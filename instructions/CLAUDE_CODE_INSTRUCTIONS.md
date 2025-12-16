# INSTRUCTIONS FOR CLAUDE CODE OPUS
## Walk-Forward Validation Implementation

**Priority:** CRITICAL - Make or Break Phase  
**Constraints:** NO HACKS, NO WORKAROUNDS, ACCEPT WHAT DATA TELLS US

---

## ✅ DATA STATUS: READY

**BTCUSDT_ml_data.parquet:**
```
Rows:   54,984 (~6.3 years)
Range:  2019-09-08 to 2025-12-15
Status: ✅ READY FOR WALK-FORWARD VALIDATION
```

**Note:** 36 rows have NaN in OHLCV (dataset boundaries) - handled by pipeline.

---

## 🚀 EXECUTION (Single Command)

```bash
cd /workspace/fund-learner
python scripts/walk_forward_validation.py
```

That's it. The script handles everything:
1. Loads full dataset
2. Trains fresh model for each period
3. Evaluates on test periods
4. Generates summary report

---

## 📋 TEST PERIODS

| Period | Test Range | Min Train | Primary |
|--------|------------|-----------|---------|
| COVID | 2020-03 to 2020-05 | 4,300 | No (bonus) |
| May 2021 | 2021-05 to 2021-07 | 14,500 | Yes |
| Luna/3AC | 2022-05 to 2022-07 | 23,000 | Yes |
| FTX | 2022-11 to 2023-01 | 27,500 | Yes |
| ETF Rally | 2024-01 to 2024-03 | 38,000 | Yes |

---

## ⚙️ FROZEN CONFIGURATION (DO NOT MODIFY)

### Model Config
```python
model_type = 'two_stage'
d_model = 64
n_heads = 4
n_encoder_layers = 2
dropout = 0.2
window_size = 64
batch_size = 32
learning_rate = 5e-5
epochs = 100
patience = 25
trade_threshold = 0.55
```

### Labeling Config
```python
strong_move_threshold = 0.010    # 1.0%
weak_move_threshold = 0.004      # 0.4%
clean_path_mae_threshold = 0.010 # 1.0%
```

---

## ❌ FORBIDDEN ACTIONS

1. **NO hyperparameter tuning** between periods
2. **NO normalizing test data** with test statistics
3. **NO transfer learning** from previous periods
4. **NO cherry-picking** which periods to report
5. **NO modifying thresholds** after seeing results

---

## ✅ SUCCESS CRITERIA

**PASS (Grade B or better):**
- ≥3 of 4 primary periods profitable
- Average Sharpe ratio > 0.8
- No period with return < -20%

**FAIL:**
- <2 periods profitable
- OR any period < -30%

---

## 📁 OUTPUT STRUCTURE

```
experiments/walk_forward/
├── period_0_covid/
│   ├── model.pt
│   ├── test_results.json
│   └── predictions.csv
├── period_1_may2021/
├── period_2_luna/
├── period_3_ftx/
├── period_4_etf/
└── summary_report.json   # Final verdict here
```

---

## 🔧 FILE TO CREATE

**scripts/walk_forward_validation.py** - Main validation script

Template provided in:
- `/home/claude/walk_forward_validation.py` (ready to use)

Copy to workspace:
```bash
cp /home/claude/walk_forward_validation.py /workspace/fund-learner/scripts/
```

---

## 🎯 EXECUTION ORDER

```
1. python scripts/walk_forward_validation.py
   └─ Trains & evaluates all 5 periods
   └─ ~30-90 minutes runtime
   
2. Review experiments/walk_forward/summary_report.json
   └─ Check verdict (PASS/FAIL)
   └─ Analyze per-period results
```

---

## 📊 EXPECTED OUTCOME SCENARIOS

**SCENARIO A: Strong Generalization (Best Case)**
- 4/4 periods profitable
- Average return: +15-25% per 3-month period
- Sharpe: 1.5-2.5
- → Proceed to paper trading

**SCENARIO B: Good Generalization (Likely)**
- 3/4 periods profitable
- Average return: +10-15% per 3-month period
- → Proceed with caution

**SCENARIO C: Partial Generalization**
- 2/4 periods profitable
- → Investigate which regimes work

**SCENARIO D: Poor Generalization**
- 0-1 periods profitable
- → Model overfit, needs redesign

---

## 🔍 DEBUGGING CHECKLIST

If validation fails unexpectedly:

1. [ ] Check train candle counts increase across periods
2. [ ] Verify no NaN in predictions
3. [ ] Confirm both long AND short trades in each period
4. [ ] Check that test periods don't overlap with training
5. [ ] Verify normalization stats came from training only

---

## 📺 EXPECTED CONSOLE OUTPUT

```
============================================================
WALK-FORWARD VALIDATION
============================================================
Dataset loaded: 54,984 rows
Dropped 36 rows with NaN in OHLCV
Clean rows: 54,948
✅ Data validated: 54,948 clean candles from 2019-09-08 to 2025-12-15

======================================================================
PERIOD: COVID Crash
Description: COVID-19 market crash - LIMITED TRAINING DATA
======================================================================
  Train: 4,XXX candles (2019-09-08 to 2020-02-29)
  Test:  2,XXX candles (2020-03-01 to 2020-05-31)
  ...training...
  Results:
    Total Return:   +XX.XX%
    Trades:         XXX
    ...

[Repeats for each period]

============================================================
WALK-FORWARD VALIDATION SUMMARY
============================================================
Period                    Return     Sharpe    Trades    Win%     Status
--------------------------------------------------------------------------------
COVID Crash               +XX.XX%     X.XX       XXX     XX.X%    BONUS
May 2021 Crash            +XX.XX%     X.XX       XXX     XX.X%    PRIMARY ✅
...

VERDICT:
Grade:          B
Passed:         YES ✅
Reasoning:      Promising: 3/4 periods profitable with good risk metrics
```

---

**END OF INSTRUCTIONS**

Copy these files to your workspace and execute in order.
