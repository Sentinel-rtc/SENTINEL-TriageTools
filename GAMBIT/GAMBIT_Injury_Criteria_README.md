# Generalized Acceleration Model for Brain Injury Threshold (GAMBIT)

The **Generalized Acceleration Model for Brain Injury Threshold (GAMBIT)** is a biomechanical criterion designed to assess the likelihood of brain injury by combining the effects of **translational** (linear) and **rotational** accelerations during a head impact event across all three spatial directions (X, Y, Z). Introduced by James A. Newman in 1985, GAMBIT provides a unified framework for evaluating head injury risk in contexts such as automotive crashes, leveraging classical engineering principles of combined stress analysis.

GAMBIT defines a threshold boundary where brain injury becomes "unacceptable" if the combined acceleration exceeds a critical value. The model has been tested against limited experimental data from cadavers, animals, and accident scenarios, with a simplified version proposed based on available evidence.

---

## Formula for GAMBIT Calculation

The original GAMBIT formulation is:

$$ G(t) = \left[ \left( \frac{a(t)}{a_c} \right)^n + \left( \frac{\alpha(t)}{\alpha_c} \right)^m \right]^{1/s} $$

Where:
- **$$\( G(t) \)$$**: GAMBIT value at time $$\( t \)$$ (unitless).
- **$$\( a(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2} \)$$**: Instantaneous resultant translational acceleration (in $$\( G \)$$, where $$\( 1 \, G = 9.81 \, m/s^2 \)$$).
- **$$\( \alpha(t) = \sqrt{\alpha_x(t)^2 + \alpha_y(t)^2 + \alpha_z(t)^2} \)$$**: Instantaneous resultant rotational acceleration (in $$\( rad/s^2 \)$$).
- **$$\( a_c \)$$**: Critical translational acceleration threshold (e.g., 250 $$\( G \)$$).
- **$$\( \alpha_c \)$$**: Critical rotational acceleration threshold (e.g., 10,000 $$\( rad/s^2 \)$$).
- **$$\( n, m, s \)$$**: Empirical exponents chosen to fit data:
  - $$\( n = m = s = 1 \)$$: Linear weighting (G1, a straight-line boundary).
  - $$\( n = m = s = 2 \)$$: Elliptical weighting (G2, a curved boundary).

The simplified GAMBIT, based on maximum resultant values and linear weighting, is:

$$ G = \frac{a_m}{250} + \frac{\alpha_m}{10,000} \leq 1 $$

Where:
- **$$\( a_m = \max(\sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}) \)$$**: Maximum resultant translational acceleration (in $$\( G \)$$).
- **$$\( \alpha_m = \max(\sqrt{\alpha_x(t)^2 + \alpha_y(t)^2 + \alpha_z(t)^2}) \)$$**: Maximum resultant rotational acceleration (in $$\( rad/s^2 \)$$).
- **$$\( G \leq 1 \)$$**: Indicates no "unacceptable" injury; $$\( G > 1 \)$$ suggests injury.

The simplified form uses $$\( a_c = 250 \, G \)$$ and $$\( \alpha_c = 10,000 \, rad/s^2 \)$$ as critical thresholds, derived from limited data and consistent with helmet evaluation standards.
The Python code for GAMBIT employs a linear weighting approach.
---

## Background

GAMBIT was developed to address the limitations of earlier head injury criteria, which often focused solely on translational acceleration (e.g., the Head Injury Criterion, HIC). By incorporating rotational acceleration and considering all three spatial directions, GAMBIT acknowledges that brain injuries, such as diffuse axonal injury (DAI) or concussion, often result from combined inertial loading in a 3D field. The model draws an analogy to engineering failure criteria under combined axial and shear stresses, treating resultant translational and rotational accelerations as "equivalent loads" that contribute to brain tissue failure.

First presented in 1985, GAMBIT has been evaluated using data from cadaver tests, animal experiments (e.g., rhesus monkeys, piglets), and theoretical considerations. While data limitations prevent precise boundary definition, the model offers a practical tool for safety system design, such as in automotive crash testing.

---

## Data Requirements for GAMBIT Calculation

To compute GAMBIT in a 3D context, the following data is required:

1. **Translational Accelerations** ($$\( a_x, a_y, a_z \)$$): Linear accelerations along the X, Y, and Z axes, typically measured with triaxial accelerometers (in $$\( G \)$$ or $$\( m/s^2 \)$$).
2. **Rotational Accelerations** ($$\( \alpha_x, \alpha_y, \alpha_z \)$$): Angular accelerations about the X, Y, and Z axes, derived from accelerometer clusters or kinematic analysis (in $$\( rad/s^2 \)$$).
3. **Time Series Data**: Continuous monitoring of all six acceleration components over the impact duration (optional for the simplified version, which uses maximum resultant values).
4. **Critical Thresholds** ($$\( a_c, \alpha_c \)$$): Predefined limits (e.g., 250 $$\( G \)$$ and 10,000 $$\( rad/s^2 \)$$) based on experimental or extrapolated data, assumed isotropic for simplicity.

Data is often collected from anthropomorphic test devices (ATDs) like the Hybrid III dummy, cadaver experiments, or scaled animal models equipped with 3D instrumentation.

---

## Interpretation and Usage

- **GAMBIT Value**: 
  - $$\( G \leq 1 \)$$: The head remains within the "safe" region, suggesting no unacceptable brain injury.
  - $$\( G > 1 \)$$: The combined resultant accelerations exceed the threshold, indicating a potential brain injury.
- **Boundary Shape**: The linear form (G1) creates a triangular safe region in the $$\( a_m \)$$-$$\( \alpha_m \)$$ plane, while the elliptical form (G2) suggests a curved boundary. Limited data supports both, with the linear form being simpler and conservative.
- **Applications**: 
  - **Automotive Safety**: Evaluates crash protection systems (e.g., airbags, seat belts) using full 3D kinematics.
  - **Helmet Design**: Assesses head protection in sports or military contexts with multi-axis impacts.
  - **Injury Research**: Provides a framework for studying combined inertial effects on the brain in three dimensions.

---

## Workflow Explanation

The following steps outline the process for calculating GAMBIT using 3D data.

### 1. Input Data and Configuration
- **Sampling Frequency**: Time interval between data points (e.g., 0.001 s for 1 kHz).
- **Acceleration Data**: Time series of $$\( a_x, a_y, a_z, \alpha_x, \alpha_y, \alpha_z \)$$.
- **Critical Thresholds**: Set $$\( a_c = 250 \, G \)$$ and $$\( \alpha_c = 10,000 \, rad/s^2 \)$$ (adjustable with new data).

### 2. Data Preprocessing
- Read acceleration data from CSV or instrumentation output.
- Filter data (e.g., 400-500 Hz Butterworth filter) to remove noise and vibrational artifacts from all six components.
- Compute resultant accelerations at each time step:
  - $$\( a(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2} \)$$.
  - $$\( \alpha(t) = \sqrt{\alpha_x(t)^2 + \alpha_y(t)^2 + \alpha_z(t)^2} \)$$.
- For simplified GAMBIT, extract maximum values $$\( a_m = \max(a(t)) \)$$ and $$\( \alpha_m = \max(\alpha(t)) \)$$.

### 3. GAMBIT Calculation
- **Full Model**: Compute $$\( G(t) \)$$ at each time step using the original equation with resultant $$\( a(t) \)$$ and $$\( \alpha(t) \)$$, adjusting $$\( n, m, s \)$$ as needed.
- **Simplified Model**: Calculate $$\( G = \frac{a_m}{250} + \frac{\alpha_m}{10,000} \)$$ using peak resultant accelerations.
- Check if $$\( G > 1 \)$$ to determine injury risk.

### 4. Output Results
- Display $$\( G \)$$ (maximum or time-varying) and injury assessment.

**Example Output**:

## Example Workflow
**Input Data** (Sample):
```csv
Time,ax,ay,az,alphax,alphay,alphaz
0.000,100,50,30,4000,5000,2000
0.001,150,70,40,6000,7000,3000
0.002,200,90,50,8000,9000,4000
```
...
**Steps**:
1. Filter Data: Apply a 400-500 Hz Butterworth filter to all six components.
Compute Resultant Maxima:
2. At each time step, calculate:
$$( a(t) = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2} ).$$ and 
$$( \alpha(t) = \sqrt{\alpha_x(t)^2 + \alpha_y(t)^2 + \alpha_z(t)^2} )$$.

For the sample data at ( t = 0.002 ):
$$( a = \sqrt{200^2 + 90^2 + 50^2} \approx 222.5 , G ).$$
$$( \alpha = \sqrt{8000^2 + 9000^2 + 4000^2} \approx 12,083 , rad/s^2 ).$$

Assume these are the maxima: $$( a_m = 222.5 , G ), ( \alpha_m = 12,083 , rad/s^2 ).$$
3. Compute GAMBIT:

$$( G = \frac{a_m}{250} + \frac{\alpha_m}{10,000} = \frac{222.5}{250} + \frac{12,083}{10,000} \approx 0.89 + 1.208 = 2.098 ).$$

4 Output: ( G = 2.098 ) (Exceeds threshold, injury likely).

GAMBIT Value: 2.098 (Exceeds threshold, injury likely)
### **Flowchart**
```plaintext

+-----------------------+
| Start                 |
+-----------------------+
         |
         v
+-----------------------+
| Load Acceleration Data|
| (ax, ay, az, αx, αy, αz) |
+-----------------------+
         |
         v
+-----------------------+
| Filter Data (400 Hz)  |
+-----------------------+
         |
         v
+-----------------------+
| Compute Resultants    |
| a(t), α(t)            |
+-----------------------+
         |
         v
+-----------------------+
| Extract Max Values    |
| (a_m, α_m)            |
+-----------------------+
         |
         v
+-----------------------+
| Compute G             |
| G = a_m/250 + α_m/10000 |
+-----------------------+
         |
         v
+-----------------------+
| Check G vs. 1         |
| Output Result         |
+-----------------------+
         |
         v
+-----------------------+
| End                   |
+-----------------------+
```

### **Reference**

1. Newman, J.A. (1986). A Generalized Acceleration Model for Brain Injury Threshold (GAMBIT). Biokinetics and Associates Limited.