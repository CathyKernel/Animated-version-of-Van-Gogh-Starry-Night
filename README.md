# Animated-version-of-Van-Gogh-Starry-Night
# Starry Night Animation Project

This project creates an artistic animation of Van Gogh's "Starry Night" painting using advanced computer vision techniques. The animation combines multiple visual effects to produce a dynamic, visually appealing result in just 8 seconds.

## Features

- 🌀 **Dynamic Swirl Effect**: Pulsing vortex motion that varies in intensity
- 🎨 **Progressive Oil Painting**: Gradually applies painterly effect
- ✨ **Starry Particles**: Random twinkling stars that appear over time
- 🌟 **Glow Effects**: Warm halos around bright areas
- 📜 **Smooth Unfolding**: Elegant reveal animation with easing
- 🌈 **Color Enhancement**: Dynamic contrast and saturation adjustments
- ⚫ **Vignette**: Subtle darkening at edges for focus

## Technical Breakdown

### Core Functions

```python
def create_dynamic_swirl_flow_field(shape, progress, max_strength=0.15, radius_ratio=0.8):
    # Creates time-varying swirl effect with pulsating radius and strength
    # Uses sine functions for smooth transitions
