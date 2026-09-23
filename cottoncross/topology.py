"""Zhang-Suen thinning, crossing-number joints and deterministic local port matching."""
from functools import lru_cache
import math
import cv2
import numpy as np
from scipy import ndimage as ndi


def zhang_suen(binary):
    im=np.pad((binary>0).astype(np.uint8),1)
    while True:
        deleted=0
        for phase in (0,1):
            p=im[1:-1,1:-1]
            n=[im[:-2,1:-1],im[:-2,2:],im[1:-1,2:],im[2:,2:],
               im[2:,1:-1],im[2:,:-2],im[1:-1,:-2],im[:-2,:-2]]
            count=sum(n)
            transitions=sum(((n[k]==0)&(n[(k+1)%8]==1)).astype(np.uint8) for k in range(8))
            if phase==0:
                cond=(n[0]*n[2]*n[4]==0)&(n[2]*n[4]*n[6]==0)
            else:
                cond=(n[0]*n[2]*n[6]==0)&(n[0]*n[4]*n[6]==0)
            remove=(p==1)&(count>=2)&(count<=6)&(transitions==1)&cond
            deleted+=int(remove.sum());p[remove]=0
        if deleted==0: break
    return im[1:-1,1:-1]


def branch_count(sk):
    im=np.pad(sk.astype(np.uint8),1)
    n=[im[:-2,1:-1],im[:-2,2:],im[1:-1,2:],im[2:,2:],
       im[2:,1:-1],im[2:,:-2],im[1:-1,:-2],im[:-2,:-2]]
    return sum(((n[k]==0)&(n[(k+1)%8]>0)).astype(np.uint8) for k in range(8))*sk


def prune_spurs(sk,max_length=6):
    """Remove only short endpoint-to-junction twigs; preserve isolated short lines."""
    sk=sk.copy()
    for _ in range(2):
        counts=branch_count(sk)
        remove=[]
        for y,x in np.argwhere(counts==1):
            path=[(int(y),int(x))];seen=set(path)
            for step in range(max_length+1):
                y,x=path[-1]
                if len(path)>1 and counts[y,x]>=3:
                    if len(path)-1<=max_length:remove.extend(path[:-1])
                    break
                nexts=[]
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        q=(y+dy,x+dx)
                        if q in seen or not (0<=q[0]<sk.shape[0] and 0<=q[1]<sk.shape[1]) or not sk[q]:continue
                        if dy and dx and (sk[y+dy,x] or sk[y,x+dx]):continue
                        nexts.append(q)
                if len(nexts)!=1:break
                path.append(nexts[0]);seen.add(nexts[0])
        if not remove:break
        for p in remove:sk[p]=0
    return sk


def match_ports(cost,unmatched=.9,margin_threshold=.12):
    """Exact minimum-cost matching of ALL ports in one joint, with an unmatched option.

    This is local joint optimization, not a globally optimal whole-image path solver.
    Keep two best distinct solutions to expose ambiguities. Costs are uncalibrated.
    """
    n=len(cost)
    if n>12:
        return dict(pairs=[],unmatched=list(range(n)),cost=float(n*unmatched),margin=0.,ambiguous=True,reason='too_many_ports')
    @lru_cache(None)
    def solve(state):
        if not state: return ((0.,(),()),)
        i=state[0];rest=state[1:];solutions=[]
        for c,p,u in solve(rest): solutions.append((c+unmatched,p,(i,)+u))
        for j in rest:
            if not np.isfinite(cost[i,j]):continue
            tail=tuple(k for k in rest if k!=j)
            for c,p,u in solve(tail):solutions.append((c+float(cost[i,j]),((i,j),)+p,u))
        return tuple(sorted(solutions,key=lambda a:(a[0],a[1],a[2]))[:2])
    result=solve(tuple(range(n)));best=result[0]
    margin=float(result[1][0]-best[0]) if len(result)>1 else float(n*unmatched)
    return dict(pairs=[list(p) for p in best[1]],unmatched=list(best[2]),cost=best[0],
                margin=margin,ambiguous=margin<margin_threshold,reason='low_margin' if margin<margin_threshold else 'resolved')


def ordered_component(coords):
    """Walk a nonbranching component. Residual branches are flagged, never silently flattened."""
    pixels={tuple(p) for p in coords.tolist()}
    neighbors={}
    for y,x in pixels:
        ns=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                q=(y+dy,x+dx)
                if (not dy and not dx) or q not in pixels:continue
                if dy and dx and ((y+dy,x) in pixels or (y,x+dx) in pixels):continue
                ns.append(q)
        neighbors[(y,x)]=sorted(ns)
    if any(len(v)>2 for v in neighbors.values()):return None,False
    ends=sorted(p for p,ns in neighbors.items() if len(ns)<=1)
    closed=not ends
    start=ends[0] if ends else min(pixels)
    path=[start];seen={start}
    while True:
        ns=[p for p in neighbors[path[-1]] if p not in seen]
        if not ns:break
        path.append(ns[0]);seen.add(ns[0])
    if len(path)!=len(pixels):return None,False
    return np.array(path,dtype=int),closed


def analyze(binary,gray=None,joint_prob=None,moments=None,radius=4,min_length=8,angle_limit=40,
            method='optimal',seed=42):
    sk=prune_spurs(zhang_suen(binary))
    # Pixel crossing-number is a 3x3 configuration test; adjacent joint pixels merge.
    candidates=(branch_count(sk)>=3)
    expanded=ndi.binary_dilation(candidates,iterations=radius)
    # A narrow-angle overlap often creates two nearby Y vertices instead of a single X.
    # Learned crossing peaks join these vertices inside one local crossing region.
    if joint_prob is not None and candidates.any():
        peaks=(joint_prob==ndi.maximum_filter(joint_prob,size=13))&(joint_prob>.5)
        distance=ndi.distance_transform_edt(~candidates)
        seeds=np.argwhere(peaks&(distance<=10))
        for cy,cx in seeds:
            y0,y1=max(0,cy-10),min(sk.shape[0],cy+11)
            x0,x1=max(0,cx-10),min(sk.shape[1],cx+11)
            yy,xx=np.ogrid[y0:y1,x0:x1]
            expanded[y0:y1,x0:x1]|=((yy-cy)**2+(xx-cx)**2<=100)
    joint_labels,nj=ndi.label(expanded,structure=np.ones((3,3)))
    cut=sk.copy();cut[expanded]=0
    segment_labels,ns=ndi.label(cut,structure=np.ones((3,3)))
    segments=[];rejected=0
    sizes=np.bincount(segment_labels.ravel())
    for i,box in enumerate(ndi.find_objects(segment_labels),1):
        if box is None or sizes[i]<min_length:continue
        coords=np.argwhere(segment_labels[box]==i)+np.array([box[0].start,box[1].start])
        path,closed=ordered_component(coords)
        if path is None:
            rejected+=1;continue
        segments.append(dict(id=len(segments),points_yx=path,closed=closed))
    joints=[];ports_by_joint={i:[] for i in range(1,nj+1)}
    centers={}
    for j,box in enumerate(ndi.find_objects(joint_labels),1):
        if box is None:continue
        points=np.argwhere((joint_labels[box]==j)&candidates[box])+np.array([box[0].start,box[1].start])
        if not len(points):continue
        centers[j]=points.mean(axis=0)
    for seg in segments:
        if seg['closed']:continue
        for end in (0,1):
            path=seg['points_yx'] if end==0 else seg['points_yx'][::-1]
            p=path[0];y,x=p
            nearby=joint_labels[max(0,y-2):y+3,max(0,x-2):x+3]
            labels=np.unique(nearby[nearby>0])
            if not len(labels):continue
            j=int(min(labels,key=lambda q:np.linalg.norm(p-centers[q])))
            k=min(12,len(path)-1)
            # Outward tangent: direction leaving the crossing, estimated over several pixels.
            v=path[k].astype(float)-path[0];v/=max(np.linalg.norm(v),1e-6)
            intensity=float(np.mean(gray[path[:k+1,0],path[:k+1,1]]))/255 if gray is not None else 0.
            ports_by_joint[j].append(dict(segment=seg['id'],end=end,point_yx=p.tolist(),direction_yx=v.tolist(),intensity=intensity))
    links={};rng=np.random.default_rng(seed)
    for j,center in centers.items():
        ports=ports_by_joint[j];n=len(ports)
        costs=np.full((n,n),np.inf)
        for a in range(n):
            for b in range(a+1,n):
                if ports[a]['segment']==ports[b]['segment']:continue
                va=np.array(ports[a]['direction_yx']);vb=np.array(ports[b]['direction_yx'])
                deviation=abs(180-math.degrees(math.acos(float(np.clip(va@vb,-1,1)))))
                limit=30 if method=='greedy' else angle_limit
                if deviation>limit:continue
                appearance=abs(ports[a]['intensity']-ports[b]['intensity'])
                # Penalize a proposed bridge crossing low-confidence foreground (when available via binary).
                pa=np.array(ports[a]['point_yx']);pb=np.array(ports[b]['point_yx'])
                q=np.rint(np.linspace(pa,pb,20)).astype(int)
                support=float(np.mean(binary[q[:,0],q[:,1]]>0))
                orientation_cost=0.
                if moments is not None:
                    cy,cx=np.rint(center).astype(int)
                    theta=np.arctan2(va[0],va[1])
                    descriptor=np.array([np.cos(2*theta),np.sin(2*theta),np.cos(4*theta),np.sin(4*theta)])
                    orientation_cost=float(np.mean((moments[:,cy,cx]-descriptor)**2))
                costs[a,b]=costs[b,a]=deviation/limit if method=='greedy' else deviation/limit+.25*appearance+.2*(1-support)+.1*orientation_cost
        if method=='greedy':
            available=list(rng.permutation(n));pairs=[]
            while available:
                a=int(available.pop())
                options=[int(b) for b in available if np.isfinite(costs[a,b])]
                if options:
                    b=min(options,key=lambda b:costs[a,b]);available.remove(b);pairs.append([a,b])
            matched={p for pair in pairs for p in pair}
            solution=dict(pairs=pairs,unmatched=[i for i in range(n) if i not in matched],margin=None,ambiguous=False,reason='greedy_baseline')
        else:solution=match_ports(costs)
        accepted=[] if solution['ambiguous'] else solution['pairs']
        for a,b in accepted:
            ka=(ports[a]['segment'],ports[a]['end']);kb=(ports[b]['segment'],ports[b]['end'])
            links[ka]=(kb,j);links[kb]=(ka,j)
        probability=float(joint_prob[tuple(np.rint(center).astype(int))]) if joint_prob is not None else None
        joints.append(dict(id=j,xy=center[::-1].tolist(),degree=n,ports=ports,solution=solution,
                           accepted_pairs=accepted,network_score=probability,
                           type='crossing_candidate' if n>=4 else 'branch_or_occlusion_candidate'))
    # Reconstruct ordered polylines by traversing endpoint links; retain unmatched ends and loops.
    fibers=[];visited=set()
    ordered_ids=sorted(range(len(segments)),key=lambda i: ((i,0) in links and (i,1) in links,i))
    for sid in ordered_ids:
        if sid in visited:continue
        entry=0 if (sid,0) not in links else 1 if (sid,1) not in links else 0
        current=sid;path=[];sids=[];closed=segments[sid]['closed']
        while current not in visited:
            visited.add(current);sids.append(current)
            points=segments[current]['points_yx']
            if entry==1:points=points[::-1]
            if path:
                # Explicit short bridge; its pixels are geometric interpolation, not observations.
                bridge=np.linspace(path[-1],points[0],max(2,int(np.linalg.norm(np.array(path[-1])-points[0]))+1))
                path.extend(bridge[1:-1].tolist())
            path.extend(points.tolist())
            nxt=links.get((current,1-entry))
            if nxt is None:break
            (current,entry),_=nxt
            if current in visited:closed=True;break
        arr=np.array(path,dtype=float)
        length=float(np.linalg.norm(np.diff(arr,axis=0),axis=1).sum())
        if closed and len(arr)>1:length+=float(np.linalg.norm(arr[-1]-arr[0]))
        fibers.append(dict(id=len(fibers),segment_ids=sids,points_xy=arr[:,::-1].tolist(),
                           length_pixels=length,closed=closed))
    result=dict(joints=joints,fibers=fibers,segment_count=len(segments),rejected_branched_components=rejected,
                crossing_candidates=sum(j['degree']>=4 for j in joints),
                unresolved_joints=sum(j['solution']['ambiguous'] for j in joints),
                coordinate_system='ROI pixels, x right y down; projected 2D geometry',
                method=method,seed=seed)
    return sk,result
