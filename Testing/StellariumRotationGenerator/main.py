# main.py
# Runs the coordinate generator and Stellarium script generator with CLI arguments.

import argparse
import os
from coordinates_generator import generate_coordinates
from script_generator import generate_stellarium_script

def main():
    parser = argparse.ArgumentParser(
        description="Generate Stellarium JS script from coordinates."
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to coordinates text file. If not given, a new one will be generated."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="stellarium_script.ssc",
        help="Output Stellarium script file (default: stellarium_script.ssc)"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=2,
        help="Delay before screenshot in seconds (default: 2)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg"],
        default="png",
        help="Screenshot format (png or jpg)"
    )

    # Coordinate generator settings (only used if -i is not given)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sphere", "plane", "band"],
        default="plane",
        help="Coordinate generation mode (sphere, plane, band)"
    )
    parser.add_argument(
        "--ra-step",
        type=int,
        default=45,
        help="Step size for RA in degrees"
    )
    parser.add_argument(
        "--dec-step",
        type=int,
        default=45,
        help="Step size for Dec in degrees (only sphere/band)"
    )
    parser.add_argument(
        "--dec-fixed",
        type=int,
        default=0,
        help="Dec value for plane mode"
    )
    parser.add_argument(
        "--dec-min",
        type=int,
        default=-30,
        help="Minimum Dec for band mode"
    )
    parser.add_argument(
        "--dec-max",
        type=int,
        default=30,
        help="Maximum Dec for band mode"
    )

    args = parser.parse_args()

    # Step 1: Get or generate coordinates
    if args.input:
        coords_file = args.input
        if not os.path.exists(coords_file):
            print(f"Error: Input file '{coords_file}' does not exist.")
            return
        print(f"Using existing coordinates file: {coords_file}")
        coords = None  # we won't have them in memory unless parsed again
    else:
        coords_file, coords = generate_coordinates(
            mode=args.mode,
            ra_step=args.ra_step,
            dec_step=args.dec_step,
            dec_fixed=args.dec_fixed,
            dec_min=args.dec_min,
            dec_max=args.dec_max,
            output_file="coords_generated.txt"
        )

    # Step 2: Generate Stellarium script
    stellarium_file, used_coords = generate_stellarium_script(
        input_file=coords_file,
        delay_before_screenshot=args.delay,
        screenshot_format=args.format,
        output_file=args.output
    )

    # Step 3: Summary
    print("\nProcess complete!")
    print(f"Coordinates file: {coords_file}")
    print(f"Stellarium script: {stellarium_file}")
    print(f"Total coordinates: {len(used_coords)}")

if __name__ == "__main__":
    main()
