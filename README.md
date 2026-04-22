# 🚢 SmartCAPEX AI - Integrated Maritime Retrofit Decision Support

**SmartCAPEX AI** is a state-of-the-art, multi-agent desktop application engineered for analyzing maritime retrofit investments. Designed specifically for the maritime sector (including aging coaster vessels), it moves beyond traditional rule-based software by leveraging a network of specialized **Intelligent Agents**. The system utilizes modern computational methods including **Emotional Artificial Neural Networks (EANN)**, **NVIDIA Modulus**, **PointNet++**, and multi-criteria decision making (MCDM) as established by *Aljahdali et al. (2025)* and *Nguyen et al. (2025)*.

---

## 🏗️ Architecture: The 4 Intelligent Agents

The core of SmartCAPEX AI operates on a decentralized, agent-based architecture where each "Agent" focuses on a highly specific domain of the maritime digital twin workflow.

### 🤖 1. The Predictor Agent (`EANN Core`)
The central "brain" of the operation. It creates the vessel's *Digital Twin* and calculates instantaneous fuel consumption by assessing physical characteristics and environmental conditions.
*   **Model:** Emotional Artificial Neural Network (EANN) based on *Aljahdali et al. (2025)*.
*   **Unique Feature (Hormonal Modulation):** Unlike standard ANNs, this agent responds to environmental "stress" factors (Wave height, Wind power) using hormonal weighting parameters ($H_a, H_b, H_c$). 
*   **Drift Detection:** Actively warns the user if external conditions exceed the safety bounds of its training distribution (e.g., extreme hurricane conditions).
*   **Extensions:** Integrates seamlessly with `PointNetAgent` for processing raw 3D hull geometry (Point Clouds) and `ModulusAgent` for high-fidelity physics AI surrogates.

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
2.  **Timeline Setup:** The `Investment Strategist` constructs a 20-year financial timeline.
3.  **Annual Iteration:** For every single year on the timeline:
    *   `Climate Guardian` determines that year's specific environmental penalties.
    *   `Predictor Agent` ingests the modified penalties along with the vessel's 3D Point Cloud to predict exact fuel consumption.
    *   `Investment Strategist` compiles fuel costs, ETS carbon taxes, and freight revenues to update the annual **Cash Flow**.
4.  **Verdict:** The system plots the 3 competing scenarios (Do-Nothing vs. Retrofit vs. New Build) dynamically and recommends the "Most Profitable Investment" directly on the UI dashboard.

---

## ⚙️ Advanced Engineering Modules

In addition to the 4-Agent core, the SmartCAPEX codebase includes state-of-the-art simulation layers:
*   **Geometry Engine (`core/geometry/FFDHullMorpher.py`):** Uses Free-Form Deformation (FFD) to generate mathematically solid B-Spline hulls, easily convertible into Point Clouds or STLs.
*   **AI-Physics Surrogates (`agents/modulus_agent.py` & `pointnet_agent.py`):** Implements NVIDIA Modulus and PointNet++ network paradigms allowing deep learning networks to "understand" hull topologies natively.
*   **Fluid Engine (`gui/cfd_widget.py`):** Visualizes localized real-time hydrodynamics utilizing Python-integrated fluid environments (`fluid-engine-dev` PyJet).
*   **OpenFOAM Bridge (`agents/openfoam_bridge.py`):** Connects the lightweight parametric interfaces to heavy-duty OpenFOAM instances for Ground-Truth CFD validation.

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

*   **Emotional Artificial Neural Networks:** *Aljahdali et al. (2025)* - Introduction of hormonal stress parameterization.
*   **Multicriteria Optimization:** *Nguyen et al. (2025)* - TOPSIS Pareto boundary definitions for maritime investments.
*   **Physics-Informed Surrogates:** Adapted from modern GPU-accelerated computing methodologies (e.g., FNO, NVIDIA).
