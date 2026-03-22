def calculate_resistance(bands):
    if len(bands) < 3: return "Unknown"

    # Direction Check
    if bands[0]['color'] in ['GOLD', 'SILVER']:
        print("  -> Reading backwards? Reversing bands...")
        bands.reverse()

    tolerance = "20%"  # Default if no tolerance band
    last_band = bands[-1]
    calc_bands = bands

    if last_band['color'] in ['GOLD', 'SILVER']:
        if last_band['color'] == 'GOLD': tolerance = "5%"
        if last_band['color'] == 'SILVER': tolerance = "10%"
        calc_bands = bands[:-1]

    if len(calc_bands) < 2: return "Error"

    try:
        digit1 = calc_bands[0]['val']
        digit2 = calc_bands[1]['val']

        # Logic การหา Multiplier
        if len(calc_bands) == 2:
            # ถ้าเจอแค่ 2 แถบ (+Tolerance)
            # เคส Red-Red-(Brownหาย)-Gold = 220 Ohm
            multiplier_idx = 1
        else:
            multiplier_idx = calc_bands[2]['val']

        multiplier = 10 ** multiplier_idx if multiplier_idx >= 0 else (0.1 if multiplier_idx == -1 else 0.01)
        total_ohms = ((digit1 * 10) + digit2) * multiplier

        if total_ohms >= 1e6:
            return f"{total_ohms / 1e6:.2f}M Ohms {tolerance}"
        elif total_ohms >= 1e3:
            return f"{total_ohms / 1e3:.2f}k Ohms {tolerance}"
        else:
            return f"{total_ohms:.1f} Ohms {tolerance}"
    except:
        return "Calc Error"