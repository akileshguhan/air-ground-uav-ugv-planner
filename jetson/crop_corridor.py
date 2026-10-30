import sys, math, numpy as np
A = sys.path[0]
sys.path.insert(0, A + '/ugv-nav'); sys.path.insert(0, A + '/ugv-nav/scripts')
from ugvnav.d455_ingest import read_ply_xyz
from run_real_ply import crop_corridor
src, dst = sys.argv[1], sys.argv[2]
p = read_ply_xyz(src)
u = np.array([math.cos(math.radians(45)), math.sin(math.radians(45))])
c = crop_corridor(p, np.array([55.0, 20.0]), u, 30.0, 6.0, margin=3.0).astype('<f4')
hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\n"
       "property float y\nproperty float z\nend_header\n" % len(c)).encode()
open(dst, 'wb').write(hdr + c.tobytes())
print("%d of %d points kept -> %s" % (len(c), len(p), dst))
