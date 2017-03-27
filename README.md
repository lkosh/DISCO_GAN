# DISCO_GAN
Course project titled " Generative neural network models for human pose estimation". The project is based on an article by D. Bouchacourt, M. P. Kumar, S. Nowozin, "DISCO Nets: DISsimilarity COefficient Networks", NIPS 2016


* utils/scores.py : define here additional scoring function if needed. We have implemented the \alpha -\beta norm with \alpha = 2 (as used in our experiment)
* examples/HandPoseEstimation/train.py : launching script
* examples/HandPoseEstimation/hand_pose.py : defines the DISCO Nets and runs training
* examples/HandPoseEstimation/hand_pose_testing.py : testing utils specific to hand pose estimation
* GAN.py - Generative adversarial network as described in the paper. Work is still in progress
