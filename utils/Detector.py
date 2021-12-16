import glob

import dlib
import os


class Detecting():
    def __init__(self, faces_folder, predictor, output_path):
        self._face_folder = faces_folder
        self._predictor_path = predictor
        self._output_path = output_path

    def gui(self):
        predictor = dlib.shape_predictor(self._predictor_path)
        detector = dlib.get_frontal_face_detector()
        print("Showing detections and predictions on the images in the faces folder...")
        # win = dlib.image_window()

        for f in glob.glob(os.path.join(self._face_folder, "*.jpg")):
            print("Processing file: {}".format(f))
            img = dlib.load_rgb_image(f)
            write_file = f.replace(f'{self._face_folder}/', '').replace('.jpg', '')
            # win.clear_overlay()
            # win.set_image(img)

            dets = detector(img, 1)
            if len(dets) > 0:
                print("Number of faces detected: {}".format(len(dets)))
                for k, d in enumerate(dets):
                    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                        k, d.left(), d.top(), d.right(), d.bottom()))
                    # Get the landmarks/parts for the face in box d.
                    shape = predictor(img, d)
                    print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                              shape.part(1)))
                    # win.add_overlay(shape)
                    if not os.path.exists(self._output_path):
                        os.mkdir(self._output_path)
                    with open(f'{self._output_path}/{write_file}.pts', 'w') as file:
                        file.write('version: 1\nn_points: 68\n{\n')
                        for i in range(68):
                            file.write(str(shape.part(i)).replace('(', '').replace(')', '').replace(',', '') + '\n')
                        file.write('}')
                    file.close()

                # win.add_overlay(dets)
                # dlib.hit_enter_to_continue()


class Acc(Detecting):
    def __init__(self, faces_folder, predictor, Xml_path):
        super().__init__(faces_folder, predictor, Xml_path)
        self.output_path = Xml_path

    def model_acc(self):
        print(f'Test of {self._predictor_path} started..')
        print(f'Testing values {self.output_path}')
        print("\nTesting accuracy: {}".format(
            dlib.test_shape_predictor(self.output_path, self._predictor_path)))


    def train_model(self):
        options = dlib.shape_predictor_training_options()
        options.num_test_splits = 50
        options.oversampling_amount = 8
        options.oversampling_translation_jitter = 0.1
        options.cascade_depth = 25

        options.nu = 0.15
        options.tree_depth = 4
        options.be_verbose = True
        options.feature_pool_size = 400

        dlib.train_shape_predictor(self.output_path, self._predictor_path, options)
