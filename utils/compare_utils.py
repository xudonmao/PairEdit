from PIL import Image, ImageDraw, ImageFont

def create_collage(images, titles, output_path=None, images_per_row=5, padding=10, font_path=None, font_size=24):
    max_img_width = max(img.width for img in images)
    max_img_height = max(img.height for img in images)
    
    total_images = len(images)
    rows = (total_images + images_per_row - 1) // images_per_row
    
    collage_width = images_per_row * max_img_width + (images_per_row - 1) * padding
    collage_height = rows * (max_img_height + font_size + padding) + (rows - 1) * padding
    
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255))
    draw = ImageDraw.Draw(collage)
    
    x_offset = 0
    y_offset = 0
    for idx, (img, title) in enumerate(zip(images, titles)):
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_x = x_offset + (max_img_width - text_width) // 2

        draw.text((text_x, y_offset), title, fill=(0, 0, 0), font=font)
        
        img_y = y_offset + font_size + padding
        collage.paste(img, (x_offset, img_y))
        
        x_offset += max_img_width + padding
        
        if (idx + 1) % images_per_row == 0:
            x_offset = 0
            y_offset += max_img_height + font_size + padding * 2
    
    if output_path is not None:
        collage.save(output_path)
        print(f"Image save to: {output_path}")
    return collage

def concatenate_images_with_caption(img1, img2, caption, output_path, padding=10, font_path=None, font_size=36):
    width = max(img1.width, img2.width)
    height = img1.height + img2.height + padding

    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    caption_lines = caption.split('\n')
    
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in caption_lines)
    total_height = height + text_height + padding * (len(caption_lines) + 2)

    new_image = Image.new('RGB', (width, total_height), (255, 255, 255))
    
    new_image.paste(img1, (0, 0))
    
    new_image.paste(img2, (0, img1.height + padding))
    
    draw = ImageDraw.Draw(new_image)
    y_text = height + padding
    for line in caption_lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, y_text), line, fill=(0, 0, 0), font=font)
        y_text += font_size + padding 
    
    new_image.save(output_path)
    print(f"Image save to: {output_path}")

def create_collage_images_with_caption(img1, caption, output_path=None, padding=10, font_path=None, font_size=36):
    width = img1.width
    height = img1.height + padding

    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default(font_size)
    
    caption_lines = caption.split('\n')
    
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    text_height = sum(draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in caption_lines)
    total_height = height + text_height + padding * (len(caption_lines) + 2)

    new_image = Image.new('RGB', (width, total_height), (255, 255, 255))
    
    new_image.paste(img1, (0, 0))
    
    draw = ImageDraw.Draw(new_image)
    y_text = height + padding
    for line in caption_lines:
        text_width = draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0]
        text_x = (width - text_width) // 2
        draw.text((text_x, y_text), line, fill=(0, 0, 0), font=font)
        y_text += font_size + padding
    
    if output_path is not None:
        new_image.save(output_path)
        print(f"Image save to: {output_path}")
    return new_image