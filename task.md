# SmartCAPEX AI — TensorFlow → PyTorch Migration

## Task Breakdown

- [x] Analyze codebase and identify gaps between PROJECT_SUMMARY.md specs and current code
- [x] Create implementation plan
- [x] **Phase 1: Surrogate Modeler Migration** ([agents/surrogate_modeler.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/surrogate_modeler.py))
  - [x] Rewrite EANN model from Keras to PyTorch (`nn.Module`)
  - [x] Rewrite EmotionalLearningRate callback as a PyTorch LR scheduler
  - [x] Rewrite PyQtProgressCallback as a PyTorch training hook
  - [x] Rewrite [train_models()](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/surrogate_modeler.py#250-350) with PyTorch training loop
  - [x] Rewrite [predict()](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/surrogate_modeler.py#351-384) with `torch.no_grad()` inference
  - [x] Add `load_ship_d_dataset()` method (Ship-D CSV ingestion)
  - [x] Add Drift Detection logic
  - [x] Save/load `.pt` weights instead of [.keras](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/models/pinn_navier_stokes.keras)
- [x] **Phase 2: PINN CFD Agent Migration** ([agents/pinn_cfd_agent.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/pinn_cfd_agent.py))
  - [x] Rewrite [NavierStokesPINN](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/pinn_cfd_agent.py#45-161) from `keras.Model` to `nn.Module`
  - [x] Rewrite physics loss with `torch.autograd.grad`
  - [x] Rewrite training loop with PyTorch optimizer
  - [x] Rewrite [predict()](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/surrogate_modeler.py#351-384) / [solve_instant()](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/agents/pinn_cfd_agent.py#368-423) with torch tensors
  - [x] Save/load `.pt` weights instead of [.keras](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/models/pinn_navier_stokes.keras)
- [x] **Phase 3: Entry Point & Dependencies**
  - [x] Update [main_gui.py](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/main_gui.py) to remove TensorFlow import
  - [x] Update [requirements.txt](file:///c:/Users/abdur/Desktop/Scap/SmartCAPEX_AI_KIM_0202/requirements.txt): replace `tensorflow` with `torch`
- [x] **Phase 4: Verification**
  - [x] Test surrogate modeler training (headless)
  - [x] Test PINN CFD agent training (headless)
  - [x] 18/18 checks passed ✓
