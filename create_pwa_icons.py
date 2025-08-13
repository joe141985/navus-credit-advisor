#!/usr/bin/env python3
"""
Simple PWA icon generator for NAVUS
Creates PNG icons from scratch using PIL/Pillow
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Define colors
    BRAND_COLOR = "#667eea"
    WHITE = "#ffffff"
    GOLD = "#FFD700"
    
    def hex_to_rgb(hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_icon(size, output_path):
        """Create a NAVUS icon at specified size"""
        # Create image with transparent background
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw background circle
        margin = 0
        draw.ellipse([margin, margin, size-margin, size-margin], 
                    fill=hex_to_rgb(BRAND_COLOR))
        
        # Calculate proportions
        card_width = int(size * 0.5)
        card_height = int(size * 0.33)
        card_x = (size - card_width) // 2
        card_y = int(size * 0.375)
        
        # Draw credit card
        draw.rounded_rectangle([card_x, card_y, card_x + card_width, card_y + card_height], 
                             radius=int(size * 0.04), fill=hex_to_rgb(WHITE))
        
        # Draw card stripe
        stripe_height = int(size * 0.04)
        stripe_y = card_y + int(card_height * 0.25)
        draw.rectangle([card_x, stripe_y, card_x + card_width, stripe_y + stripe_height], 
                      fill=hex_to_rgb(BRAND_COLOR))
        
        # Draw chip
        chip_size = int(size * 0.08)
        chip_x = card_x + int(size * 0.06)
        chip_y = stripe_y + stripe_height + int(size * 0.04)
        draw.rounded_rectangle([chip_x, chip_y, chip_x + chip_size, chip_y + int(chip_size * 0.75)], 
                             radius=int(size * 0.01), fill=hex_to_rgb(GOLD))
        
        # Try to add text (fallback if font not available)
        try:
            # Try to use a system font
            font_size = int(size * 0.125)
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
        
        # Draw NAVUS text
        text = "NAVUS"
        text_y = card_y + card_height + int(size * 0.08)
        
        # Get text dimensions
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
        except:
            text_width = len(text) * (font_size // 2)  # Fallback estimate
        
        text_x = (size - text_width) // 2
        draw.text((text_x, text_y), text, fill=hex_to_rgb(WHITE), font=font)
        
        # Save the image
        img.save(output_path, 'PNG')
        print(f"Created icon: {output_path}")
    
    # Create directory if it doesn't exist
    output_dir = "/Users/joebanerjee/NAVUS/chat-frontend"
    
    # Generate icons in required sizes
    sizes_to_create = [
        (192, "icon-192x192.png"),
        (512, "icon-512x512.png"),
        (180, "apple-touch-icon.png"),  # iOS
        (32, "favicon-32x32.png"),
        (16, "favicon-16x16.png"),
        (144, "icon-144x144.png"),      # Android
        (96, "icon-96x96.png"),
        (72, "icon-72x72.png"),
        (48, "icon-48x48.png")
    ]
    
    for size, filename in sizes_to_create:
        output_path = os.path.join(output_dir, filename)
        create_icon(size, output_path)
    
    print("\n✅ All PWA icons created successfully!")
    print("Icons created:")
    for size, filename in sizes_to_create:
        print(f"  - {filename} ({size}x{size})")

except ImportError:
    print("❌ PIL/Pillow not available. Installing...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("✅ Pillow installed. Please run the script again.")
    except:
        print("❌ Could not install Pillow. Please install manually: pip install Pillow")
except Exception as e:
    print(f"❌ Error creating icons: {e}")