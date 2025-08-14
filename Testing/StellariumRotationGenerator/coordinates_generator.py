# coordinates_generator.py
# Generates coordinates (RA, Dec) for different scanning patterns.

def generate_coordinates(
    mode="plane",
    ra_step=45,
    dec_step=45,
    dec_fixed=0,
    dec_min=-30,
    dec_max=30,
    output_file="coords_generated.txt"
):
    coords = []

    if mode == "sphere":
        # RA: 0 → <360
        # Dec: -90 → 90
        for dec in range(-90, 91, dec_step):
            for ra in range(0, 360, ra_step):
                coords.append((float(ra), float(dec)))

    elif mode == "plane":
        # Single constant Dec
        for ra in range(0, 360, ra_step):
            coords.append((float(ra), float(dec_fixed)))

    elif mode == "band":
        # Limited Dec range
        for dec in range(dec_min, dec_max + 1, dec_step):
            for ra in range(0, 360, ra_step):
                coords.append((float(ra), float(dec)))

    else:
        raise ValueError("Invalid mode. Use 'sphere', 'plane', or 'band'.")

    # Save as simple comma-separated values
    with open(output_file, "w", newline="") as f:
        for ra, dec in coords:
            f.write(f"{ra}, {dec},\n")

    print(f"Generated {len(coords)} coordinates and saved to {output_file}")
    return output_file, coords
