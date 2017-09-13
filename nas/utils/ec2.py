#!/usr/bin/env python

"""
    ec2.py
    
    Launch EC2 instance and run training
"""

import json
import boto3
import base64
import uuid

# --
# Helpers

client = boto3.client('ec2')

def make_cmd(configs):
    cmd = b"""#!/bin/bash

echo '
#!/bin/bash

MID=$1
CONFIG=$2

PATH="/home/ubuntu/.anaconda/bin:/home/ubuntu/.local/bin:$PATH"

sudo nvidia-smi -pm ENABLED -i 0
sudo nvidia-smi -ac 2505,875 -i 0

cd /home/ubuntu/projects/pcifar/nas/

sudo git stash
sudo git pull

mkdir -p logs
python ./grid-point.py \
    --config-b64 $CONFIG \
    --epochs 2 \
    --run ec2 2>./logs/$MID.log || true

export AWS_SHARED_CREDENTIALS_FILE=/home/ubuntu/.aws/credentials
find results/ | fgrep $MID | xargs -I {} aws s3 cp {} s3://cfld-nas/{}
aws s3 cp ./logs/$MID.log s3://cfld-nas/logs/$MID.log
' >> /home/ubuntu/run.sh

chmod +x /home/ubuntu/run.sh

    """
    
    for config in configs:
        cmd += """
# su -c '/home/ubuntu/run.sh %s %s' -m ubuntu
        """ % (config['model_name'], base64.b64encode(json.dumps(config)))
    
    return cmd, base64.b64encode(cmd)


def launch_spot(cmd_b64, spot_price=0.20):
    return client.request_spot_instances(
        DryRun=False,
        SpotPrice=str(spot_price),
        InstanceCount=1,
        Type='one-time',
        LaunchSpecification={
            'ImageId'        : 'ami-f3a74389', # pcifar-gpu-v2
            'KeyName'        : 'rzj',
            'SecurityGroups' : ['ssh'],
            'InstanceType'   : 'p2.xlarge',
            'Placement' : {
                'AvailabilityZone': 'us-east-1c', # cheapest when I looked
            },
            'UserData' : cmd_b64
        }
    )

# --
# Run

# Define job
mid = str(uuid.uuid1())
config = {
    "op_keys": ["double_bnconv_3", "identity", "add"], 
    "red_op_keys": ["conv_1", "double_bnconv_3", "add"],
    "model_name" : mid,
}

cmd_clear, cmd_b64 = make_cmd([config])
print cmd_clear
launch_spot(cmd_b64)
