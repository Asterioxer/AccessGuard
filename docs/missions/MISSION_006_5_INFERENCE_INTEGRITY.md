# Mission 006.5 — Inference Failure Integrity & Scenario Matrix

**Repository:** `Asterioxer/AccessGuard`  
**Working Branch:** `feature/unified-inference-evidence-integrity`  
**PR:** [#1 — Mission 006.2: Unified Inference Architecture & Evidence Integrity](https://github.com/Asterioxer/AccessGuard/pull/1)  
**Date:** 2026-11-19  
**Status:** VALIDATED  

---

## 1. Objective

To eliminate the silent conversion of inference exceptions into false-positive successful execution records. This matrix documents the exact input conditions, required lifecycle states, and empirical test verification results across all specialist model adapters and the unified inference service.

---

## 2. Scenario Verification Matrix

| Scenario ID | Test Scenario Description | Input Condition | Expected Status | Expected Execution Type | Actual Status | Actual Execution Type | Test Name | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **YOLO Successful Inference with Detections** | Valid 256x256 BGR image, loaded model, 1 ground hazard detected | `READY` | `model_inference` | `READY` | `model_inference` | `test_scenario_a_yolo_successful_inference_with_detections` | **PASSED** |
| **B** | **YOLO Successful Inference with Zero Detections** | Valid 256x256 BGR image, loaded model, clean unblocked pavement | `READY` | `model_inference` | `READY` | `model_inference` | `test_scenario_b_yolo_successful_inference_with_zero_detections` | **PASSED** |
| **C** | **YOLO Inference Exception** | Model loaded; `predict()` raises internal runtime exception | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_c_yolo_inference_exception` | **PASSED** |
| **D** | **ONNX Inference Exception** | ONNX session initialized; `OrtSession::Run` raises runtime error | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_d_onnx_inference_exception` | **PASSED** |
| **E** | **Missing Model Checkpoint** | Nonexistent weight path; `allow_fallback=False` (strict policy) | `UNAVAILABLE` | `unavailable` | `UNAVAILABLE` | `unavailable` | `test_scenario_e_missing_checkpoint` | **PASSED** |
| **F** | **Explicit Development Fallback** | Nonexistent weight path; `allow_fallback=True` (dev policy) | `DEGRADED` | `heuristic_fallback` | `DEGRADED` | `heuristic_fallback` | `test_scenario_f_explicit_development_fallback` | **PASSED** |
| **G** | **Hardware Resource Exhaustion** | CUDA Out-of-Memory / system RAM exhaustion exception | `RESOURCE_EXHAUSTED` | `failed` | `RESOURCE_EXHAUSTED` | `failed` | `test_scenario_g_resource_exhaustion` | **PASSED** |
| **H-1** | **Invalid Input: NoneType** | Input array is `None` | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_h_invalid_image_input[None]` | **PASSED** |
| **H-2** | **Invalid Input: Empty Array** | Input array has shape `(0,)` | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_h_invalid_image_input[invalid_input1]` | **PASSED** |
| **H-3** | **Invalid Input: Zero Dimensions** | Input array has shape `(0, 0, 3)` | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_h_invalid_image_input[invalid_input2]` | **PASSED** |
| **H-4** | **Invalid Input: String Non-Array** | Input is string `"not_an_array"` | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_h_invalid_image_input[not_an_array]` | **PASSED** |
| **I** | **Impossible Contract Enforcement** | Constructing impossible combinations (`UNAVAILABLE + model_inference`) | `ValueError` raised | N/A | `ValueError` raised | N/A | `test_scenario_i_impossible_contract_combinations_rejected` | **PASSED** |
| **J** | **End-to-End Service Failure Integrity** | Single-adapter request to `ImageProcessingService` with forced YOLO failure | `overall_status="FAILED"` | `failed` | `overall_status="FAILED"` | `failed` | `test_scenario_j_e2e_service_integrity_with_forced_failure` | **PASSED** |
| **K** | **Zero-Detection End-to-End Integrity** | Single-adapter request to `ImageProcessingService` with 0 detections | `overall_status="SUCCESS"` | `model_inference` | `overall_status="SUCCESS"` | `model_inference` | `test_scenario_k_zero_detection_distinguishable_from_failure` | **PASSED** |
| **L-1** | **Sidewalk Validator Missing Checkpoint** | Missing `dinov2_validator.pt` checkpoint | `UNAVAILABLE` | `unavailable` | `UNAVAILABLE` | `unavailable` | `test_scenario_l_sidewalk_validator_failure_reporting` | **PASSED** |
| **L-2** | **Sidewalk Validator Runtime Crash** | Forward pass raises runtime exception | `INFERENCE_FAILED` | `failed` | `INFERENCE_FAILED` | `failed` | `test_scenario_l_sidewalk_validator_failure_reporting` | **PASSED** |

---

## 3. Scientific Invariant Verification

1. **Zero-Detection Independence:**
   A scene with zero detected barriers is not classified as an error or failure. When genuine inference completes without exception, `AdapterInferenceResult.status == READY` and `execution_type == "model_inference"` with `output_count == 0` and `errors == []`.
2. **Exception Containment:**
   Exceptions thrown during tensor processing, memory allocation, or ONNX evaluation are caught at the adapter boundary, strictly classified, and encapsulated into `AdapterInferenceResult` with `execution_type="failed"`, `execution_succeeded=False`, and `execution_attempted=True`.
3. **No Silent Model Substitution:**
   If ONNX fails during session execution, the pipeline does not silently retry on PyTorch or fall back to an unverified heuristic mock under strict genuine execution policy.
