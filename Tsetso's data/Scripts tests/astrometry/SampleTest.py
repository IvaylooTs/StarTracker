import astrometry
with astrometry.Solver(
    astrometry.series_5200.index_files(
        cache_directory="astrometry_cache",
        scales={4,5,6},    
        )
) as solver:

    stars = [
        [959.47, 539.61], 
        [1878.00, 404.50],
        [312.00, 610.50],
        [818.86, 839.27],
        [48.12, 744.20],
        [668.63, 581.63],
        [519.38, 713.62],
        [45.28, 668.94],
        [69.71, 524.31],
        [997.68, 779.32],
        [1059.28, 987.44],
        [1346.90, 521.10],
        [767.17, 291.70],
        [1133.85, 121.50],
        [353.96, 12.16],
        [858.17, 751.07],
        [22.63, 714.57],
        [1092.18, 251.61],
        [1121.52, 897.48],
        [1387.62, 527.38],
    ]

    solution = solver.solve(
        stars=stars,
        size_hint=None,
        position_hint=None,
        solution_parameters=astrometry.SolutionParameters(),
    )

    if solution.has_match():
        print(f"{solution.best_match().center_ra_deg=}")
        print(f"{solution.best_match().center_dec_deg=}")
        print(f"{solution.best_match().scale_arcsec_per_pixel=}")