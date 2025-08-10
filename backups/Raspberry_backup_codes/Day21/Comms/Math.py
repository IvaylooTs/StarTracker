# math.py
normalization_threshold = 1e-4


def inverse_quaternion(w, x, y, z):
    norm_sq = w**2 + x**2 + y**2 + z**2
    if norm_sq == 0:
        return 1, 0, 0, 0  # Avoid division by zero
    return (
        w / norm_sq,
        -x / norm_sq,
        -y / norm_sq,
        -z / norm_sq
    )

def is_normalized(w, x, y, z):
    norm = w**2 + x**2 + y**2 + z**2
    return abs(norm - 1.0) < normalization_threshold  # Check if norm is close to 1.0
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)

def round_quaternion(q, decimals=6):
    return tuple(round(c, decimals) for c in q)