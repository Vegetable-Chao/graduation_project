#python train_cifar.py --save '511first_s3' --arch 'second_s3' 2>&1 | tee 511second_s3.txt
#python train.py --save '511second_s3' --arch 'second_s3' 2>&1 | tee 511second_s3.txt        
python train_cifar.py --arch 'second_s3'