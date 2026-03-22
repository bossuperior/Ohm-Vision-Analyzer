def calculate_resistance(bands):
    # ป้องกัน List ว่าง หรืออ่านมาได้น้อยกว่า 3 แถบ
    if not bands or len(bands) < 3: 
        return None, "Error: Insufficient bands"

    processed_bands = list(bands)

    # Direction Check: ถ้าแถบแรกเป็น Tolerance (Gold/Silver) แปลว่าอ่านกลับหลัง
    if processed_bands[0]['color'] in ['GOLD', 'SILVER']:
        processed_bands = processed_bands[::-1]

    tolerance = "20%" # Default
    last_band = processed_bands[-1]
    
    # เช็กว่าแถบสุดท้ายเป็น Tolerance หรือไม่
    if last_band['color'] in ['GOLD', 'SILVER', 'BROWN', 'RED'] and len(processed_bands) > 3:
        if last_band['color'] == 'GOLD': tolerance = "5%"
        elif last_band['color'] == 'SILVER': tolerance = "10%"
        elif last_band['color'] == 'BROWN': tolerance = "1%"
        elif last_band['color'] == 'RED': tolerance = "2%"
        
        calc_bands = processed_bands[:-1]
    else:
        calc_bands = processed_bands

    try:
        # แบ่งเคส 4-band vs 5-band
        if len(calc_bands) == 3:
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']
            multiplier_idx = calc_bands[2]['val']
            base_val = (digit1 * 10) + digit2
            
        elif len(calc_bands) == 4:
            digit1 = calc_bands[0]['val']
            digit2 = calc_bands[1]['val']
            digit3 = calc_bands[2]['val']
            multiplier_idx = calc_bands[3]['val']
            base_val = (digit1 * 100) + (digit2 * 10) + digit3
            
        else:
            return None, f"Error: Invalid band format ({len(calc_bands)} data bands)"

        # คำนวณค่า R รวม 
        multiplier = 10 ** multiplier_idx
        total_ohms = float(base_val * multiplier) # แปลงเป็น Float ให้ชัวร์ว่าคำนวณต่อได้

        # Format string ให้สวยงาม
        if total_ohms >= 1e6:
            display_str = f"{total_ohms / 1e6:.2f}M Ohms {tolerance}"
        elif total_ohms >= 1e3:
            display_str = f"{total_ohms / 1e3:.2f}k Ohms {tolerance}"
        else:
            display_str = f"{total_ohms:.1f} Ohms {tolerance}"
            
        # 🔥 ส่งค่ากลับไปทั้ง 2 รูปแบบ (ตัวเลข, ข้อความ)
        return total_ohms, display_str
            
    except KeyError:
        return None, "Error: Missing 'val' or 'color' key"
    except Exception as e:
        return None, f"Calc Error: {str(e)}"