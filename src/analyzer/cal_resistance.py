def calculate_resistance(bands):
    # ป้องกัน List ว่าง หรืออ่านมาได้น้อยกว่า 3 แถบ (ใช้งานไม่ได้แน่นอน)
    if not bands or len(bands) < 3: 
        return "Error: Insufficient bands"

    # สร้าง Copy ของ list เพื่อป้องกันการเผลอแก้ข้อมูลต้นฉบับ
    processed_bands = list(bands)

    # Direction Check: ถ้าแถบแรกเป็น Tolerance (Gold/Silver) แปลว่าอ่านกลับหลัง
    if processed_bands[0]['color'] in ['GOLD', 'SILVER']:
        processed_bands = processed_bands[::-1]

    tolerance = "20%" # Default
    last_band = processed_bands[-1]
    
    # เช็กว่าแถบสุดท้ายเป็น Tolerance หรือไม่ (ปกติ 4 แถบมักเป็น Gold/Silver, 5 แถบอาจเป็น Brown/Red)
    # สมมติว่าในดิกชันนารีมี key 'is_tolerance' หรือเช็กจากสี
    if last_band['color'] in ['GOLD', 'SILVER', 'BROWN', 'RED'] and len(processed_bands) > 3:
        if last_band['color'] == 'GOLD': tolerance = "5%"
        elif last_band['color'] == 'SILVER': tolerance = "10%"
        elif last_band['color'] == 'BROWN': tolerance = "1%"
        elif last_band['color'] == 'RED': tolerance = "2%"
        
        # ตัดแถบ Tolerance ออกไปเพื่อคำนวณตัวเลข
        calc_bands = processed_bands[:-1]
    else:
        # กรณีไม่มีแถบ Tolerance ชัดเจน (เช่น ถ่ายติดแค่ 3 แถบ)
        calc_bands = processed_bands

    try:
        # แบ่งเคส 4-band vs 5-band (ถ้าเหลือ 3 แถบ แปลว่าเดิมคือ 4 แถบสี)
        if len(calc_bands) == 3:
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']
            multiplier_idx = calc_bands[2]['val']
            base_val = (digit1 * 10) + digit2
            
        # ถ้าเหลือ 4 แถบ แปลว่าเดิมคือ 5 แถบสี
        elif len(calc_bands) == 4:
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']
            digit3 = calc_bands[2]['val']
            multiplier_idx = calc_bands[3]['val']
            base_val = (digit1 * 100) + (digit2 * 10) + digit3
            
        else:
            return f"Error: Invalid band format ({len(calc_bands)} data bands)"

        # คำนวณค่า R รวม (Python รองรับ 10**-1 และ 10**-2 อัตโนมัติ)
        multiplier = 10 ** multiplier_idx
        total_ohms = base_val * multiplier

        # Format string ให้สวยงาม
        if total_ohms >= 1e6:
            return f"{total_ohms / 1e6:.2f}M Ohms {tolerance}"
        elif total_ohms >= 1e3:
            return f"{total_ohms / 1e3:.2f}k Ohms {tolerance}"
        else:
            return f"{total_ohms:.1f} Ohms {tolerance}"
            
    except KeyError:
        return "Error: Missing 'val' or 'color' key in dictionary"
    except Exception as e:
        return f"Calc Error: {str(e)}"