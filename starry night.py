import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import random

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def apply_flow(img, flow):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    flow_x = np.clip(x + flow[:, :, 0], 0, w - 1)
    flow_y = np.clip(y + flow[:, :, 1], 0, h - 1)
    return cv2.remap(img, flow_x.astype(np.float32), flow_y.astype(np.float32), cv2.INTER_CUBIC)

def create_dynamic_swirl_flow_field(shape, progress, max_strength=0.15, radius_ratio=0.8):
    h, w = shape[:2]
    radius = int(min(h, w) * radius_ratio * (0.5 + 0.5 * np.sin(progress * np.pi * 2)))
    
    y, x = np.ogrid[:h, :w]
    dx = x - w // 2
    dy = y - h // 2
    distance = np.sqrt(dx**2 + dy**2)
    
    # Dynamic strength based on progress
    strength = max_strength * (0.5 + 0.5 * np.sin(progress * np.pi * 4))
    
    angle = strength * (distance / (radius + 1e-5)) * 2 * np.pi
    mask = distance < radius
    angle[~mask] = 0
    
    flow_x = dx * np.cos(angle) - dy * np.sin(angle) + w // 2 - x
    flow_y = dx * np.sin(angle) + dy * np.cos(angle) + h // 2 - y
    
    return np.dstack((flow_x, flow_y))

def enhance_colors_dynamically(img, progress):
    # Varying enhancement based on progress
    contrast = 1.0 + 0.5 * np.sin(progress * np.pi * 2)
    saturation = 1.2 + 0.3 * np.sin(progress * np.pi * 3)
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX)
    l = np.clip(l * contrast, 0, 255).astype(np.uint8)
    a = np.clip(a * saturation, 0, 255).astype(np.uint8)
    b = np.clip(b * saturation, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def advanced_oil_painting(img, radius=2, levels=12):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    quantized = (gray // (256 // levels)) * (256 // levels)
    
    h, w = img.shape[:2]
    result = np.zeros_like(img)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            region = quantized[i-radius:i+radius+1, j-radius:j+radius+1]
            hist = np.bincount(region.ravel(), minlength=256)
            dominant = np.argmax(hist)
            
            mask = (gray[i-radius:i+radius+1, j-radius:j+radius+1] == dominant)
            if mask.any():
                for c in range(3):
                    color_region = img[i-radius:i+radius+1, j-radius:j+radius+1, c]
                    result[i,j,c] = np.mean(color_region[mask])
            else:
                result[i,j] = img[i,j]
    
    return result

def create_starry_particles(img, progress, density=0.003, max_size=3):
    h, w = img.shape[:2]
    particles = np.zeros_like(img)
    
    # Only add particles after 20% progress
    if progress < 0.2:
        return img
    
    num_particles = int(density * h * w * (progress - 0.2) / 0.8)
    
    for _ in range(num_particles):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        size = random.randint(1, max_size)
        brightness = random.uniform(1.5, 3.0)
        color = np.array([brightness * 255, brightness * 220, brightness * 180])  # Warm star color
        
        cv2.circle(particles, (x, y), size, color, -1, lineType=cv2.LINE_AA)
    
    return cv2.addWeighted(img, 1.0, particles, 0.7, 0)

def add_glow_effect(img, progress, intensity=0.3):
    if progress < 0.3:
        return img
    
    # Create glow based on bright areas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (0, 0), 10)
    mask = mask.astype(np.float32) / 255.0
    
    # Yellowish glow
    glow = np.zeros_like(img)
    glow[:,:,0] = 50  * mask  # Blue (less)
    glow[:,:,1] = 150 * mask  # Green
    glow[:,:,2] = 255 * mask  # Red
    
    # Vary intensity with progress
    current_intensity = intensity * (1 + np.sin(progress * np.pi * 4)) / 2
    return cv2.addWeighted(img, 1.0, glow.astype(np.uint8), current_intensity, 0)

def smooth_unfold_effect(img, progress):
    h, w = img.shape[:2]
    unfolded = img.copy()
    
    # Smooth unfolding with easing function
    eased_progress = np.sin(progress * np.pi / 2)  # Ease-in function
    center_y = int(h * eased_progress)
    
    if center_y < h:
        x = np.linspace(0, w - 1, w)
        y = np.linspace(center_y, h - 1, h - center_y)
        xx, yy = np.meshgrid(x, y)
        
        # Dynamic curve based on progress
        curve_strength = 30 * (1 - eased_progress) * (0.5 + 0.5 * np.sin(progress * np.pi * 4))
        zz = curve_strength * np.sin(xx / w * np.pi * 2)
        
        remapped = cv2.remap(
            img[center_y:],
            xx.astype(np.float32),
            (yy + zz - center_y).astype(np.float32),
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT
        )
        unfolded[center_y:] = remapped
    
    return unfolded

# Main execution
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
img = load_image(url)
height, width = img.shape[:2]

# Resize for better performance and artistic effect
new_width = 1024  # Higher resolution for better quality
new_height = int(height * (new_width / width))
img = cv2.resize(img, (new_width, new_height))
height, width = img.shape[:2]

# Video writer with higher quality
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('enhanced_starry_night.mp4', fourcc, 30.0, (width, height))

duration_seconds = 8  # Shorter duration
total_frames = duration_seconds * 30

for frame in range(total_frames):
    progress = frame / total_frames
    
    # 1. Smooth unfolding effect
    unfolded = smooth_unfold_effect(img, progress)
    
    # 2. Dynamic swirl effect (more pronounced in middle section)
    if progress < 0.85:
        flow = create_dynamic_swirl_flow_field(unfolded.shape, progress)
        swirled = apply_flow(unfolded, flow)
    else:
        swirled = unfolded
    
    # 3. Dynamic color enhancement
    color_enhanced = enhance_colors_dynamically(swirled, progress)
    
    # 4. Progressive oil painting effect
    if progress > 0.4:
        effect_strength = min(1.0, (progress - 0.4) / 0.6)
        oil_painted = advanced_oil_painting(color_enhanced)
        final = cv2.addWeighted(color_enhanced, 1 - effect_strength, 
                               oil_painted, effect_strength, 0)
    else:
        final = color_enhanced
    
    # 5. Add star particles (appears after unfolding)
    final = create_starry_particles(final, progress)
    
    # 6. Add glow effect to bright areas
    final = add_glow_effect(final, progress)
    
    # 7. Final subtle vignette effect
    if progress > 0.2:
        vignette = np.ones((height, width, 3), dtype=np.float32)
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xx, yy = np.meshgrid(x, y)
        vignette_strength = 0.3 * (1 - np.sqrt(xx**2 + yy**2) / np.sqrt(2))
        vignette = np.clip(vignette - vignette_strength[..., np.newaxis], 0, 1)
        final = (final.astype(np.float32) * vignette).astype(np.uint8)
    
    out.write(final)
    
    if frame % 10 == 0:
        print(f"Processing frame {frame}/{total_frames} ({progress*100:.1f}%)")

out.release()
print("Enhanced animation complete!")
