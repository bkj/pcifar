#!/usr/bin/env python

"""
    ec2.py
    
    Launch EC2 instance for running training
"""

import sys
import uuid
import json
import boto3
import base64
import argparse
import numpy as np
from uuid import uuid4
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-jobs', type=int, required=True)
    parser.add_argument('--n-workers', type=int, required=True)
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-schedule', type=str, default='linear')
    return parser.parse_args()

# --
# Helpers

client = boto3.client('ec2')

def make_cmd(configs, epochs=20, lr_schedule='cyclical', terminate=False):
    
    print >> sys.stderr, len(configs)
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
    --epochs %d \
    --lr-schedule %s \
    --train-history \
    --run ec2 2>./logs/$MID.log || true

export AWS_SHARED_CREDENTIALS_FILE=/home/ubuntu/.aws/credentials
find results/ | fgrep $MID | xargs -I {} aws s3 cp {} s3://cfld-nas/{}
aws s3 cp ./logs/$MID.log s3://cfld-nas/logs/$MID.log

' >> /home/ubuntu/run.sh

chmod +x /home/ubuntu/run.sh

    """ % (epochs, lr_schedule)
    
    for config in configs:
        cmd += """
su -c '/home/ubuntu/run.sh %s %s' -m ubuntu
        """ % (config['model_name'], base64.b64encode(json.dumps(config)))
    
    if terminate:
        cmd += """
echo "terminating..."
sudo halt
        """
    
    return cmd, base64.b64encode(cmd)


def launch_spot(cmd_b64, spot_price=0.25):
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
            'UserData' : cmd_b64,
        }
    )


# --
# Run

if __name__ == "__main__":
    args = parse_args()
    
    configs = []
    for line in sys.stdin:
        config = json.loads(line.strip())
        
        now = datetime.now().strftime('%Y%m%dT%H%M%S')
        short_uuid4 = str(uuid4())[:8]
        config['model_name'] = '-'.join((config['model_name'], short_uuid4, now))
        configs.append(config)
    
    if len(configs) > args.max_jobs:
        raise Exception('too many configs!')
    
    config_chunks = np.array_split(configs, args.n_workers)
    
    for i, config_chunk in enumerate(config_chunks):
        print "submitting request %d" % i
        cmd_clear, cmd_b64 = make_cmd(config_chunk, terminate=True)
        print >> sys.stderr, cmd_clear
        print launch_spot(cmd_b64)

