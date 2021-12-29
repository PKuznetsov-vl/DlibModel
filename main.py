import json
from Plot.plot_graph import Plot
from utils.file_managment import FileOp, PathOp, Detector, Checker, CreateXML
from utils.Detector import Acc, Detecting

with open('config.json') as file:
    config = json.load(file)
file.close()

create_data_val =  config['create_data_val']
train_model_val = config['train_model_val']
test_predictor_val = config['create_predictor_val']
directory_for_train = config['directory_for_train']
faces_folder_for_test = config['faces_folder_for_test']
predictor = config['predictor']
xml_path = config['xml']
my_predictor_output = config['predictor_output']
orig_data_path=config['orig_data_path']
out_path=config['graph_output_path']
predictor_data_path= config['predictor_data_path']


def create_data(directory,xml_path):
    file_manage = FileOp(directory)
    images, pts, full_path = file_manage.get_files_from_dir()
    print('getting points....')
    ch = Checker(full_path)
    ph = ch.check_pts()
    print('check_faces....')
    #del ph[400:]
    img_manage = Detector(images_path=ph, directory=directory)
    values, path_values = img_manage.det()
    corr_val = []

    path_manage = PathOp(path_values, directory=directory)
    points = path_manage.get_pts()
    xml = CreateXML(path_values, values, points, xml_path)

    print('create_xml....')
    xml.create_xml()
    print('Done....')


def test_model(faces_folder, predictor, xml):
    tst = Acc(faces_folder=faces_folder, predictor=predictor, Xml_path=xml)
    tst.model_acc()


def train_model(faces_folder, predictor, xml):
    train = Acc(faces_folder=faces_folder, predictor=predictor, Xml_path=xml)
    train.train_model()


def test_predictor(faces_folder, predict, output_ph):
    train = Detecting(faces_folder=faces_folder, predictor=predict, output_path=output_ph)
    train.gui()


if __name__ == '__main__':
    if create_data_val:
        print('Create data tor training ')
        create_data(directory_for_train,xml_path)
        print('Done')
    if train_model_val:
        print('Train model ')
        train_model(faces_folder_for_test,predictor,xml_path)
        print('Done')
    elif test_predictor_val:
        print('Test predictor\n Will be created .pts files ')
        test_predictor(faces_folder_for_test,predictor,my_predictor_output)
        print('Plot... ')
        gr = Plot(gt_path=orig_data_path,predictions_path=predictor_data_path,output_path=out_path)
        gr.main()
        print('Done')