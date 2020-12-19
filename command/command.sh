## conventional KD
python student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch MobileNetV2  --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch vgg8         --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch MobileNetV2  --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch ShuffleV1    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV1    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV2    --lr 0.01 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_16_2     --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch resnet8x4    --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_40_1     --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet56      --student-arch resnet20     --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch vgg8         --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_vgg16         --student-arch vgg8         --lr 0.05 --strategy 0 --k 64 --b 64 --w 1000 --seed 0 --gpu-id 0

## UNIXKD
python student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch MobileNetV2  --lr 0.01 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch vgg8         --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_ResNet50      --student-arch MobileNetV2  --lr 0.01 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch ShuffleV1    --lr 0.01 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV1    --lr 0.01 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch ShuffleV2    --lr 0.01 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_16_2     --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet32x4    --student-arch resnet8x4    --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2      --student-arch wrn_40_1     --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_resnet56      --student-arch resnet20     --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_vgg13         --student-arch vgg8         --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0
python student_v0.py --teacher-path ./experiments/teacher_vgg16         --student-arch vgg8         --lr 0.05 --strategy 3 --k 48 --b 32 --w 10 --seed 0 --gpu-id 0

