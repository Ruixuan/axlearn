#!/bin/bash
# should be launched under the root dir of axlearn
export TENSORBOARD_PORT=6007
export PORT_AJAX_COR=7012
export PORT_NEURON_RT_ROOT_COMM_ID=7013
export OUTPUT_DIR="/mnt/task_runtime/trainer_output/"
## Set AJAX Comms Port
export BOLT_NEURON_RT_ROOT_COMM_ID="$BOLT_KUBERNETES_POD_NAME.bolt-pods.turi-bolt.svc.cluster.local:$PORT_NEURON_RT_ROOT_COMM_ID"
export BOLT_COORDINATOR_ADDRESS="$BOLT_KUBERNETES_POD_NAME.bolt-pods.turi-bolt.svc.cluster.local:$PORT_AJAX_COR"
echo "BOLT_COORDINATOR_ADDRESS: $BOLT_COORDINATOR_ADDRESS"
echo "BOLT_NEURON_RT_ROOT_COMM_ID: $BOLT_NEURON_RT_ROOT_COMM_ID"
export OMPI_COMM_WORLD_SIZE=1
export OMPI_COMM_WORLD_RANK=0
# Neuron compiler flags
#export NEURON_FRAMEWORK_DEBUG=1
export NEURON_CC_FLAGS="--framework=XLA"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --model-type transformer"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --no-internal-hlo-remat"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --distribution-strategy=llm-training"
#export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --verbose=debug"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --internal-hlo2tensorizer-options='--verify-hlo '"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --enable-mixed-precision-accumulation"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} -O1"
export NEURON_CC_FLAGS="${NEURON_CC_FLAGS} --dump=/mnt/task_runtime/dump"
# Neuron PJRT flags
export NEURON_WHILE_LOOP_UNROLL=1
export NEURON_RUN_TRIVIAL_COMPUTATION_ON_CPU=1
# Neuron runtime flags
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1
export CCOM_SOCKET_IFNAME=eth0
# Neuron env vars for distributed training
# Disable for Bolt
export devices_per_node=32
# Edited for Bolt
export COORDINATOR_ADDRESS=$BOLT_COORDINATOR_ADDRESS
# Edited for Bolt
export NEURON_RT_ROOT_COMM_ID=$BOLT_NEURON_RT_ROOT_COMM_ID
export NEURON_PJRT_PROCESSES_NUM_DEVICES=$(printf '%s,' $(seq 1 $OMPI_COMM_WORLD_SIZE | xargs -I {} echo $devices_per_node) | sed 's/,$//')
export NEURON_PJRT_PROCESS_INDEX=$OMPI_COMM_WORLD_RANK
export LD_LIBRARY_PATH="/opt/amazon/efa/lib/"
export FI_LOG_LEVEL="debug" #warn
export FI_EFA_USE_DEVICE_RDMA="1"
export FI_PROVIDER="efa"
export FI_EFA_FORK_SAFE=1
unset OMPI_MCA_orte_hnp_uri
set
mkdir -p $OUTPUT_DIR
# Run the training script
export DATA_DIR=gs://axlearn-public/tensorflow_datasets
echo "COORDINATOR_ADDRESS: $COORDINATOR_ADDRESS"
echo "OMPI_COMM_WORLD_SIZE: $OMPI_COMM_WORLD_SIZE"
echo "OMPI_COMM_WORLD_RANK: $OMPI_COMM_WORLD_RANK"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "DATA_DIR: $DATA_DIR"
echo "setup done!"
env

python3 -u -m axlearn.common.launch_trainer_main --initialization_timeout=3600 \
    --module=text.gpt.c4_trainer --config=fuji-7B-v1 \
    --trainer_dir=/mnt/task_runtime/trainer_output --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=neuron --mesh_selector=neuron-trn1n.32xlarge-32 \
    --distributed_coordinator=bolt-zeicmdr8i5-7savz45bs4.bolt-pods.turi-bolt.svc.cluster.local:7012 \
    --num_processes=1 \
    --process_id=0 2>&1 | tee ${OUTPUT_DIR}/${PMIX_HOSTNAME}.log
