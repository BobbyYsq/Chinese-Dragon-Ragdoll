#  Chinese Dragon Ragdoll

![Dragon Ragdoll Overview](media/dragon_ragdoll_overview.gif)

This project combines ideas from **Assignment 7 (Kinematics)** and  
**Assignment 8 (Mass-Spring Systems)** to create a physically-based  
**ragdoll Chinese dragon**. Only the **skeleton** is simulated with
springs and gravity; the **mesh** is deformed in real time via
linear blend skinning.
---
## Personal Information

- **Name:** Siqi Yang  
- **UtorID:** yangsi41  
- **Student Number:** 1009574667  
- **Augmented Assignments:** A7 (Kinematics) and A8 (Mass-Spring Systems)
---

## Instructions

### Project Structure (relevant parts)

```text
csc317-dragon/
  CMakeLists.txt
  main.cpp                # custom ragdoll + kinematics demo
  include/                # Skeleton, Bone, LBS headers from A7 and A8
  src/                    # original A7/A8 source files (not all used)
  data/
    dragon/
      dragon.obj          # Chinese dragon mesh
  media/                  # GIFs/videos for the Piece/Compilation Verification
  ...
````
### Dependencies

* **C++17** compiler

  * macOS: AppleClang / clang++
  * Windows: MSVC (Visual Studio 2019/2022) or a recent MinGW toolchain
* **CMake ≥ 3.20**
* **OpenGL + GLFW** (brought in by libigl)
* **libigl** (fetched automatically via CMake’s `FetchContent`)
* **Eigen3** (fetched automatically by libigl)

No extra Python tools or external runtime assets are required.

---

### Build & Run on macOS / Linux

From the project root (`csc317-dragon`):

```bash
# 1. Configure
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 2. Build
cmake --build build --config Release -j8

# 3. Run
./build/dragon_showcase ./data/dragon/dragon.obj
```

If the mesh path argument is omitted, `main.cpp` tries to use the default
relative path `../data/dragon/dragon.obj` from the executable directory,
so the safest is to **pass the explicit path** as shown above.

---

### Build & Run on Windows (Visual Studio / MSVC)

1. Open **“x64 Native Tools Command Prompt for VS 2019/2022”** (or similar).

2. Change to the project root:

   ```bat
   cd path\to\csc317-dragon
   ```

3. Configure and build:

   ```bat
   rmdir /S /Q build
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release -j8
   ```

4. Run the executable from the project root so the relative data path works:

   ```bat
   .\build\Release\dragon_showcase.exe .\data\dragon\dragon.obj
   ```

If Visual Studio opens the generated solution, it should build the same
target `dragon_showcase` with the supplied CMake configuration.

---

## Controls

### Camera (libigl defaults)

* **Rotate:** Left mouse drag
* **Pan:** Right mouse drag
* **Zoom:** Mouse scroll

### Ragdoll Interaction

* **Left-click + drag (on a green point):**
  Grab a **body joint** (along the dragon’s spine) and drag it in 3D.
  The grabbed joint turns **pink** and becomes a **pinned mass**;
  the rest of the skeleton responds via springs and gravity.

* **Space:** Toggle physics simulation **play / pause**

* **R / r:** Reset the ragdoll to the original rest pose above the floor

* **M / m:** Toggle mesh faces on/off

* **W / w:** Toggle mesh wireframe on/off

* **S / s:** Toggle skeleton visibility (bones and body joint markers)

Notes:

* Only **body joints** (centerline) are directly draggable.
* Whisker bones participate fully in the mass-spring simulation but cannot
  be grabbed; they move indirectly through the springs.
* The collision floor is **invisible** and affects only the skeleton joints,
  not the mesh.

---

## Description

### Overview

The final piece is a **single-mode ragdoll simulation** of a Chinese dragon:

* The dragon’s **skeleton** (body + left whisker + right whisker) is
  simulated as a **mass-spring system** with gravity and damping.
* The **dragon mesh** is *not* simulated; it is deformed via
  **linear blend skinning (LBS)** driven by the ragdoll skeleton.
* The user can interactively **grab and drag** body joints to make the whole
  dragon sway and settle realistically on an invisible floor.

The design intentionally combines the core ideas of:

* **A7 (Kinematics):** Skeletons, bone rest transforms, forward kinematics,
  and LBS.
* **A8 (Mass-Spring):** Spring networks, gravity, damping, and collision with
  a plane.

All of the additional logic is implemented in `main.cpp`.

---

### Key Features and Where They Are Implemented

#### 1. Procedural Skeleton From the Mesh (`main.cpp`)

**Goal:** Build a clean, straight **body chain** and symmetric **whisker chains**
directly from `dragon.obj`, without using a separate JSON rig.

* **Functions:**

  * `build_body_joints_from_samples(...)`
  * `build_symmetric_whiskers(...)`
* **Details:**

  * The body uses 12 joints (→ 11 bones) between the tail and the head.

    * Sample vertices (indices copied from Open3D’s picking log) are collected
      along the centerline.

      * A small PCA on these samples yields the main body axis.
      * Joints are evenly spaced along this axis to enforce a straight spine.
  * Each whisker uses 4 joints (→ 3 bones).

    * Right/left whisker samples near the nose are averaged to estimate
      a direction and length.
    * The direction is projected into the plane perpendicular to the body axis
      so that whiskers are not slanted along the spine.
    * The final whisker directions are mirrored across the body axis to make
      the left and right whiskers symmetric.

This corrects noisy sample data and ensures neatly aligned bones for the
body and two whiskers.

---

#### 2. Skeleton, Bone Transforms, and Skinning Weights (`main.cpp` + `include/`)

**Goal:** Convert joint positions into a `Skeleton` compatible with A7, and
compute skinning weights procedurally.

* **Function:** `build_skeleton_and_weights(...)`
* **Uses:**

  * `Skeleton` / `Bone` classes from `include/Skeleton.h` and `include/Bone.h`
  * `linear_blend_skinning(...)` from `src/linear_blend_skinning.cpp`
* **Details:**

  * For each bone:

    * The **x-axis** is aligned with the direction from tail joint to tip joint.
    * An orthonormal basis is constructed to form a `rest_T` transform.
    * Parent indices are set so that:

      * The body forms a single chain.
      * Both whiskers branch from the head bone.
  * Skinning weights `W`:

    * For each mesh vertex, distances to bone centers are computed.
    * Inverse-distance-squared weights are used.
    * Only the **top-4** bone influences per vertex are kept and normalized.

The mesh is then deformed every frame using
`linear_blend_skinning(V, skeleton, T, W, U);` with updated bone transforms `T`.

---

#### 3. Full Ragdoll Mass-Spring System (`main.cpp`)

**Goal:** Simulate all joints (body + whiskers) as a mass-spring network
subject to gravity and floor collision.

* **Structures & Functions:**

  * `struct Ragdoll`
  * `build_full_ragdoll(...)`
  * `step_ragdoll(Ragdoll & R, double dt, int substeps)`
* **Details:**

  * Each joint is a **mass point** with position and velocity.
  * Every bone becomes a **spring** between its tail and tip joints.
    Two extra springs link the head joint to each whisker root.
  * Gravity is intentionally strong (`(0, -5000, 0)`) so the dragon
    falls and settles quickly.
  * Semi-implicit Euler integration with multiple substeps is used for
    stability.
  * An invisible floor at `floor_y` (slightly below the lowest mesh vertex)
    clamps joint positions and applies a small bounce if they move downward.

Body joints are slightly heavier; whisker joints are lighter so that
whiskers flutter more when the dragon moves.

---

#### 4. Interactive “IK-like” Dragging (`main.cpp`)

**Goal:** Allow the user to grab the spine and manipulate the dragon, with
the rest of the skeleton responding physically.

* **Callback logic:**

  * `viewer.callback_mouse_down`
  * `viewer.callback_mouse_move`
  * `viewer.callback_mouse_up`
* **How it works:**

  * On mouse down, body joints are projected to screen space and the nearest
    one within a small radius is selected.
  * The selected joint becomes **pinned**:

    * Its position is forced to follow the unprojected mouse cursor
      (`rag_full.pinned_target`).
  * Other joints are free to move according to forces and springs.
  * When the mouse is released, the joint is unpinned and continues to move
    under the ragdoll simulation.

This gives a natural, intuitive way to “pose” the dragon by physically
dragging its body rather than directly editing Euler angles as in A7.

---

#### 5. Rendering and Visualization (`main.cpp`)

* The dragon mesh is drawn with faces and optional wireframe (`M` / `W` keys).
* The skeleton is visualized as:

  * Colored **points** at body joints (green when free, pink when pinned).
  * **Edges** along each bone.
* Floor is **not rendered**; only its physical effect is visible via the
  dragon’s motion.

---

## Acknowledgements

### External Model

* **Model:** *Chinese dragon*
* **Author:** MAXDESIGN-3D
* **Source:** Sketchfab
* **URL:** [https://sketchfab.com/3d-models/chinese-dragon-a5ad5ec06f43461f95bb93e95fe7553b](https://sketchfab.com/3d-models/chinese-dragon-a5ad5ec06f43461f95bb93e95fe7553b)
* **License:** Creative Commons Attribution (CC BY)

Only the **geometry** (OBJ) from this model is used.
The original rig and animation data are **not** used; instead,
this project builds a new skeleton and skinning weights on top of the mesh.

### Libraries and Starter Code

* **libigl**, **Eigen**, **GLFW**, and **glad** (pulled in via the CSC317
  assignment CMake setup).
* Starter code and infrastructure from:

  * **CSC317 Assignment 7:** Kinematics & linear blend skinning.
  * **CSC317 Assignment 8:** Mass-spring systems and plane collision
    (conceptual inspiration; the implementation here is custom and joint-based).

No additional external C++ libraries beyond those used in the course
framework are required.

