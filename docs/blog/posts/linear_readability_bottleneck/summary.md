# Linear separability vs nonlinear heads: run summary
- Linear probe accuracy moved from **0.6527** at epoch 0 to **0.9865** at epoch 20.
- Best nonlinear gain moved from **-0.0172** at epoch 0 to **+0.0016** at epoch 20.
- Spearman correlation between linear accuracy and best nonlinear gain: **0.200**. Negative values support the bottleneck story.
- Test within/between scatter ratio moved from **1.7477** to **0.0510**. Lower means tighter class clouds relative to class separation.
- At the final checkpoint, the smallest split-conformal average set size was **1.000** from **<bound method NDFrame.head of epoch                            20
head                  linear_logreg
head_family                  linear
alpha                           0.1
coverage_target                 0.9
conformal_score                rank
calibration_source          holdout
n_fit                          7500
n_selection                    2500
n_calib                        5000
n_test                         5000
selection_acc                0.9956
test_acc                     0.9876
qhat                            0.0
coverage                     0.9876
coverage_gap                 0.0876
avg_set_size                    1.0
median_set_size                 1.0
singleton_rate                  1.0
empty_rate                      0.0
valid_at_target                True
selected_overall              False
selected_nonlinear            False
fit_seconds                0.042221
error                              
Name: 30, dtype: object>**, with coverage **0.988** at target **0.90**.
- The head selected on the independent selection split was **<bound method NDFrame.head of epoch                        20
head                  small_mlp
head_family           nonlinear
alpha                       0.1
coverage_target             0.9
conformal_score            rank
calibration_source      holdout
n_fit                      7500
n_selection                2500
n_calib                    5000
n_test                     5000
selection_acc            0.9972
test_acc                 0.9888
qhat                        0.0
coverage                 0.9888
coverage_gap             0.0888
avg_set_size                1.0
median_set_size             1.0
singleton_rate              1.0
empty_rate                  0.0
valid_at_target            True
selected_overall           True
selected_nonlinear         True
fit_seconds            0.025201
error                          
Name: 35, dtype: object>**; on test it had accuracy **0.9888**, coverage **0.989**, and average set size **1.000**.

Interpretation: nonlinear heads are most informative as a diagnostic. Large positive deltas mean the frozen representation still contains label structure that is not linearly readable. Small deltas at high linear-probe accuracy mean the backbone has already done the geometric work. Conformal prediction adds a second diagnostic: at fixed coverage, did the nonlinear head actually reduce uncertainty, or did it only flip a few borderline point predictions?
