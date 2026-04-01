# Architecture

## Core machine

1. **Capture**
   - open camera
   - grab latest frame

2. **Warp**
   - AprilTag lock defines the homography
   - warp output is the working coordinate frame

3. **Trigger**
   - fixed mode: old red/image rectangles
   - dynamic mode: two stacked horizontal red zones

4. **ROI synthesis**
   - fixed mode: use saved image rectangle
   - dynamic mode: compute `x_center` from upper/lower zones, place ROI above the upper zone

5. **Optional model**
   - disabled by default in this pack

6. **UI/report**
   - camera window
   - warp window
   - red mask window
   - wrapped text console window
