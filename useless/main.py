import teeth_number_detection_yolo as yolo_model
from pbl_boundary import pbl_detection_model
from util import *
import sys

# dataset location
TEETH_DATASET, TEETH_PBL, TEETH_NUMBER = './dataset/original-data', './dataset/pbl-mask', './dataset/teeth-color'
PBL_EPOCH, PBL_BATCH = 100 , 16


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Please input the parameter. [pbl, yolo]')

    # train unet model for pbl boundary detection
    if '--pbl' in sys.argv:
        print('==> start unet traning')
        # load dataset
        original_dataset = load_dataset(TEETH_DATASET)
        pbl_dataset = load_dataset(TEETH_PBL)

        # split dataset 80% train
        (pbl_X_train, pbl_y_train), (pbl_X_test, pbl_y_test) = split_train_test(original_dataset, pbl_dataset, 80)

        # fit pbl model
        pbl_model = pbl_detection_model(input_shape = (512, 512, 1)).model()
        pbl_model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        history = pbl_model.fit(pbl_X_train,pbl_y_train, validation_split = 0.2, batch_size=PBL_BATCH, epochs=PBL_EPOCH, verbose=1)
        pbl_model.save('./result/unet/unet_model.h5')
        unet_lost_history(history)
        print('Finish UNET training...')

    # teeth number detection with yolov5
    if '--yolo' in sys.argv:
        print('==> start yolo traning')
        yolo_model = yolo_model.teeth_number_detection_yolo()
