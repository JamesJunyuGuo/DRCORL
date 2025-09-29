TASK="OfflineAntVelocityGymnasium-v1"; seed=100; python3 -u safe_behavior.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineHopperVelocityGymnasium-v1"; seed=100; python3 -u safe_behavior.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineSwimmerVelocityGymnasium-v1"; seed=100; python3 -u safe_behavior.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineWalker2dVelocityGymnasium-v1"; seed=100; python3 -u safe_behavior.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

# critic

TASK="OfflineAntVelocityGymnasium-v1"; seed=100; python3 -u safe_critic.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineHopperVelocityGymnasium-v1"; seed=100; python3 -u safe_critic.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineSwimmerVelocityGymnasium-v1"; seed=100; python3 -u safe_critic.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2

TASK="OfflineWalker2dVelocityGymnasium-v1"; seed=100; python3 -u safe_critic.py --expid ${TASK}-baseline-seed${seed} --env $TASK --seed ${seed} --beta 0.2
