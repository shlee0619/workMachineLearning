    directions = {
        "N": n - r_q,
        "S": r_q - 1,
        "E": n - c_q,
        "W": c_q - 1,
        "NE": min(n - r_q, n - c_q),
        "NW": min(n - r_q, c_q - 1),
        "SE": min(r_q - 1, n - c_q),
        "SW": min(r_q - 1, c_q - 1)
    }

    
    obstacle_positions = {(r, c) for r, c in obstacles}

    
    for (r, c) in obstacles:
        if c == c_q:
            if r > r_q:  
                directions["N"] = min(directions["N"], r - r_q - 1)
            else:  
                directions["S"] = min(directions["S"], r_q - r - 1)
        elif r == r_q:
            if c > c_q:  
                directions["E"] = min(directions["E"], c - c_q - 1)
            else:  
                directions["W"] = min(directions["W"], c_q - c - 1)
        elif r - r_q == c - c_q:
            if r > r_q:  
                directions["NE"] = min(directions["NE"], r - r_q - 1)
            else: 
                directions["SW"] = min(directions["SW"], r_q - r - 1)
        elif r - r_q == -(c - c_q):
            if r > r_q:  
                directions["NW"] = min(directions["NW"], r - r_q - 1)
            else:  
                directions["SE"] = min(directions["SE"], r_q - r - 1)

    
    return sum(directions.values())