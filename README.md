# 🚢 Retrosim - Integrated Maritime Retrofit Decision Support

**Retrosim** is a state-of-the-art, multi-agent desktop application engineered for analyzing maritime retrofit investments. Designed specifically for the maritime sector (including aging coaster vessels), it moves beyond traditional rule-based software by leveraging a network of specialized **Intelligent Agents**. The system utilizes modern computational methods including **XGBoost Surrogate Modeling**, **Geometry-Conditioned Fourier Neural Operators (GC-FNO)**, **PointNet++**, and multi-criteria decision making (MCDM) as established by *Westermann et al. (2020)* and *Nguyen et al. (2025)*.

---

## 🏗️ Architecture: The 4 Intelligent Agents

The core of Retrosim operates on a decentralized, agent-based architecture where each "Agent" focuses on a highly specific domain of the maritime digital twin workflow.

### 🤖 1. The Predictor Agent (`Surrogate Modeler & GC-FNO`)
The central "brain" of the operation. It creates the vessel's *Digital Twin* and calculates instantaneous fuel consumption by assessing physical characteristics and environmental conditions.
*   **Tabular Model:** **XGBoost Ensemble** for instant resistance predictions based on a 45-parameter Design Vector.
*   **3D Geometric Model:** **PointNet++** for direct resistance prediction from 3D Point Cloud hull topologies.
*   **Physics AI Surrogate:** **Geometry-Conditioned Fourier Neural Operator (GC-FNO)** with dual-head architecture to predict both 3D fluid dynamics `(u,v,w,p)` and scalar resistance $C_T$, strictly enforcing physics via exact boolean boundary masking.
*   **Drift Detection:** Kriging (Gaussian Process) is used to actively warn the user with confidence bounds if external conditions exceed the safety limits of the training distribution.

### 💰 2. The Investment Strategist (`MultiObjectiveOptimizer`)
Translates pure physics into financial reality. Evaluates economic feasibility and competes various investment scenarios using NPV and Discounted Cash Flow (DCF).
*   **3-Arena Analysis:**
    1.  **Do-Nothing:** Calculates the operational loss (aging penalties) if the vessel is left as-is.
    2.  **Retrofit:** Analyzes ROI and CAPEX recovery time for fitting green technologies (Flettner Rotors, Air Lubrication, etc.).
    3.  **New Build:** Computes the 20-year yield of scrapping/selling the current vessel and constructing an AI-optimized new ship.
*   **Regulatory Economics:** Automatically integrates ETS (Emissions Trading System) carbon tax projections into the financial sheet.

### 🌍 3. The Climate Guardian (`ClimateGuardian`)
The temporal analysis agent. Evaluates scenarios on a 20-year chronological timeline (2025–2050) rather than a single point in time. 
*   **Scenario Management:** Generates year-by-year sea state deterioration factors and resistance penalties.
*   **Aging Degradation:** Quantifies ship aging effects such as Hull Fouling and Engine Degradation.
*   **Verdict Matrix:** Highlights critical break-even thresholds—*"What is profitable today might incur massive resistance penalties and losses under 2035 climate conditions."*

### 🛠️ 4. The Asset Manager (`Startup Wizard`)
The bridge between the user and the predictive backend. Ensures pristine data structures before analysis begins.
*   **Data Aggregation:** Collects vessel identity, Age, DWT, and operational expenditures (OPEX).
*   **Auto-Imputation:** Employs advanced statistical regressions to deduce missing geometric inputs (e.g., L, B, T dimensions, Block Coefficient) to build a consistent dataset to feed the `Predictor Agent`.

---

## 🔄 Interaction Flow (The System Loop)

1.  **Initialization:** User defines the vessel parameters via the `Asset Manager`. Missing geometries are reconstructed via B-Spline interpolation (`FFDHullMorpher`).
2.  **Pipeline Generation:** The `RetrosimPipeline` uses the `GeometryAssembler` to create 3D Point Clouds and SDF grid tensors.
3.  **CFD Execution:** The `OpenFOAMRunner` simulates baseline hydrodynamics to feed the FNO surrogate model.
4.  **Timeline Setup:** The `Investment Strategist` constructs a 20-year financial timeline.
5.  **Annual Iteration:** For every single year on the timeline:
    *   `Climate Guardian` determines that year's specific environmental penalties.
    *   `GC-FNO Agent` ingests the modified penalties along with the vessel's 3D SDF to predict exact fuel consumption dynamically.
    *   `Investment Strategist` compiles fuel costs, ETS carbon taxes, and freight revenues to update the annual **Cash Flow**.
6.  **Verdict:** The system plots the 3 competing scenarios (Do-Nothing vs. Retrofit vs. New Build) dynamically and recommends the "Most Profitable Investment" directly on the UI dashboard.

---

## ⚙️ Advanced Engineering Modules

In addition to the 4-Agent core, the Retrosim codebase includes state-of-the-art simulation layers:
*   **Geometry Engine (`core/geometry_assembler.py`):** Complete STL-to-SDF pipeline converting parametric 45-D Design Vectors into 6-channel 3D tensors `[SDF, x, y, z, Re, Fr]`.
*   **Automated CFD (`core/openfoam_runner.py`):** Fully automated `simpleFoam` OpenFOAM pipeline for generating ground-truth RANS hydrodynamics and extracting drag coefficients.
*   **AI-Physics Surrogates (`models/gc_fno3d.py`):** Production Geometry-Conditioned Fourier Neural Operator (GC-FNO) with a dual-head architecture predicting both the full 3D flow field `[u, v, w, p]` and scalar resistance $C_T$, utilizing exact boolean boundary masking.
*   **End-to-End Orchestrator (`pipeline/orchestrator.py`):** 3-stage `RetrosimPipeline` tying together geometry generation, automated OpenFOAM CFD generation, GC-FNO training, and inference.
*   **Fluid Engine (`gui/cfd_widget.py`):** Visualizes localized real-time hydrodynamics utilizing Python-integrated fluid environments.

---

## 💻 Installation & Usage

### Prerequisites
*   OS: Windows 10/11 (Desktop Environment)
*   Python: `3.8` to `3.13`
*   PyTorch (CUDA recommended for Modulus and PointNet)
*   PyQt6

### Setup
1. Clone the repository.
2. Install standard dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the application GUI:
   ```bash
   python main_gui.py
   ```

### Quick Start
1.  **Launch the App:** You will see the standard PyQt6 Ribbon UI.
2.  **Asset Management:** Use the *New Project* or *Startup Wizard* to input your vessel parameters.
3.  **Parametric Generation:** Generate internal 3D Point Cloud geometries. 
4.  **Simulate:** Let the 4 Agents iterate through the 20-year span.
5.  **Review Dashboard:** Check the NPV charts, ETS taxes, and final the CAPEX Verdict.

---

## 🔬 Scientific Literature Base

*   **Geometry-Conditioned Physics AI:** *Li et al. (2021)* - Fourier Neural Operator for Parametric PDEs.
*   **Multicriteria Optimization:** *Nguyen et al. (2025)* - TOPSIS Pareto boundary definitions for maritime investments.
*   **Point Cloud Feature Extraction:** *Qi et al. (2017)* - PointNet++ deep hierarchical feature learning on point sets.
*   **Parametric Hull Generation:** MIT DeCoDE Lab (Ship-D) - Large scale dataset and 45-parameter B-Spline generation.

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If using this software in research, please cite:

```text
Retrosim: Intelligent Multi-Agent Architecture for Maritime Retrofit Decisions
Based on AI-Based Decision Support Systems methodology (Bocaneala et al.)
Literature: Surrogate Modeling (Westermann et al., 2020), 
            Fourier Neural Operator (Li et al., 2021),
            Pareto Optimality (Rosso et al., 2020)
```

---

**Retrosim v1.0** - Intelligent Maritime Retrofit Decision Support
© 2025 Retrosim Development Team
