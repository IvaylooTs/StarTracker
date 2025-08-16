#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Quaternion:
    x: float
    y: float
    z: float
    w: float

    def as_tuple(self) -> Tuple[float,float,float,float]:
        return (self.x, self.y, self.z, self.w)

    def normalized(self) -> 'Quaternion':
        n = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)
        if n == 0.0:
            raise ValueError("Zero-norm quaternion")
        return Quaternion(self.x/n, self.y/n, self.z/n, self.w/n)

    @staticmethod
    def from_rotation_matrix(R):
        m00,m01,m02 = R[0]
        m10,m11,m12 = R[1]
        m20,m21,m22 = R[2]
        tr = m00 + m11 + m22
        if tr > 0:
            S = math.sqrt(tr+1.0) * 2
            w = 0.25 * S
            x = (m21 - m12) / S
            y = (m02 - m20) / S
            z = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2
            w = (m21 - m12) / S
            x = 0.25 * S
            y = (m01 + m10) / S
            z = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2
            w = (m02 - m20) / S
            x = (m01 + m10) / S
            y = 0.25 * S
            z = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2
            w = (m10 - m01) / S
            x = (m02 + m20) / S
            y = (m12 + m21) / S
            z = 0.25 * S
        return Quaternion(x,y,z,w).normalized()

def mat_vec_mul(M, v):
    """Fixed matrix-vector multiplication"""
    return (
        M[0]*v + M[1]*v[1] + M[2]*v[2],
        M[1]*v + M[1][1]*v[1] + M[1][2]*v[2],
        M[2]*v + M[2][1]*v[1] + M[2][2]*v[2]
    )

def cross(a, b):
    """Fixed cross product function"""
    return (a[1]*b[2] - a[2]*b[1], 
            a[2]*b - a*b[2], 
            a*b[1] - a[1]*b)

def normalize(v):
    x,y,z = v
    n = math.sqrt(x*x + y*y + z*z)
    if n == 0:
        raise ValueError("Zero vector")
    return (x/n, y/n, z/n)

def rot_z(deg):
    r = math.radians(deg)
    c, s = math.cos(r), math.sin(r)
    return [
        [ c,-s, 0],
        [ s, c, 0],
        [ 0, 0, 1],
    ]

def matmul33(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def radec_to_gcrs_unit(ra_deg, dec_deg):
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = math.cos(dec) * math.cos(ra)
    y = math.cos(dec) * math.sin(ra)
    z = math.sin(dec)
    return (x,y,z)

def triad_from_z(z_hat):
    z_hat = normalize(z_hat)
    up = (0.0, 0.0, 1.0)
    if abs(z_hat[2]) > 0.9:
        up = (0.0, 1.0, 0.0)
    x_hat = normalize(cross(up, z_hat))
    y_hat = cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat

def jd_from_ymd_hms(year, month, day, hour, minute, second):
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year/100)
    B = 2 - A + math.floor(A/4)
    JD = math.floor(365.25*(year+4716)) + math.floor(30.6001*(month+1)) + day + B - 1524.5
    JD += (hour + minute/60 + second/3600)/24
    return JD

def gmst_deg_from_utc_yymmdd_hhmmss(year, month, day, hour, minute, second):
    """Greenwich Mean Sidereal Time in degrees from UTC"""
    JD = jd_from_ymd_hms(year, month, day, hour, minute, second)
    JD0 = math.floor(JD - 0.5) + 0.5
    H = (JD - JD0) * 24.0
    D = JD - 2451545.0
    D0 = JD0 - 2451545.0
    T = D / 36525
    GMST = 6.697374558 + 0.06570982441908 * D0 + 1.00273790935 * H + 0.000026 * (T*T)
    GMST = (GMST % 24) * 15.0
    return GMST

def ecef_to_gcrs_rotation(era_deg):
    return rot_z(era_deg)

def body_to_gcrs_quaternion_from_radec(ra_deg, dec_deg, year, month, day, hour, minute, second):
    """
    Compute quaternion that rotates body frame to point at RA/Dec coordinates
    accounting for Earth's rotation at the specified UTC time.
    """
    era_deg = gmst_deg_from_utc_yymmdd_hhmmss(year, month, day, hour, minute, second)
    Re = ecef_to_gcrs_rotation(era_deg)
    t_g = radec_to_gcrs_unit(ra_deg, dec_deg)
    bx, by, bz = triad_from_z(t_g)
    R_body_gcrs = [
        [bx[0], by[0], bz],
        [bx[1], by[1], bz[1]],
        [bx[2], by[2], bz[2]],
    ]
    ReT = [[Re[j][i] for j in range(3)] for i in range(3)]
    R_body_ecef = matmul33(ReT, R_body_gcrs)
    q = Quaternion.from_rotation_matrix(R_body_ecef)
    return q.normalized()

def run_tests():
    # Expected quaternions provided by user
    EXPECTED = {
        (0,0): (-0.4098392, 0.57599369, -0.41042687, 0.57603202),
        (180,0): (0.41006729, -0.57620801, -0.41029401, 0.57574992),
    }
    
    # Test with UTC time: 2025-08-16T12:00:00+00:00
    year, month, day, hour, minute, second = 2025, 8, 16, 12, 0, 0
    
    print(f"Tests for UTC {year}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
    print("="*60)
    
    for (ra, dec), expected in EXPECTED.items():
        q = body_to_gcrs_quaternion_from_radec(ra, dec, year, month, day, hour, minute, second)
        got = q.as_tuple()
        
        print(f"\nRA={ra}°, Dec={dec}°:")
        print(f"  Computed:  ({got[0]:.8f}, {got[1]:.8f}, {got[2]:.8f}, {got[3]:.8f})")
        print(f"  Expected:  ({expected:.8f}, {expected[1]:.8f}, {expected[2]:.8f}, {expected[3]:.8f})")
        
        diffs = tuple(g - e for g, e in zip(got, expected))
        print(f"  Difference:({diffs[0]:.8f}, {diffs[1]:.8f}, {diffs[2]:.8f}, {diffs[3]:.8f})")
        
        norm_computed = math.sqrt(sum(x*x for x in got))
        norm_expected = math.sqrt(sum(x*x for x in expected))
        print(f"  Norm - Computed: {norm_computed:.8f}, Expected: {norm_expected:.8f}")

if __name__ == "__main__":
    run_tests()
