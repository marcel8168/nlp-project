import argparse
import logging
import webbrowser
from ActiveLearning import ActiveLearning
from Gui import GUI
from SciBertClassifier import SciBertClassifier
from System import System
from TextFile import TextFile
from constants import COLLECTION_NAME, FOLDER_NAME, PATH_TO_BRAT, SUGGESTION_ANNOTATION_TYPE


def add_to_config(file_path: str, type: str, entities: list[str]):
    """
    Add the specified entity below the [entities] section in the given file.

    Arguments
    ---------
        file_path (str): Path to the file.
        type (str): Name of the section where entities should be inserted.
        entity (str): Entity to be added.
    """
    with open(file_path, 'r') as file:
        contents = file.readlines()

    section_start_index = contents.index(f'[{type}]\n')
    section_end_index = section_start_index + 1
    next_section_index = len(contents)
    for i in range(section_start_index + 1, len(contents)):
        if contents[i].startswith('['):
            next_section_index = i
            break
    del contents[section_start_index + 1:next_section_index]

    contents.insert(section_end_index, '\n')
    section_end_index += 1
    for entity in entities:
        contents.insert(section_end_index, entity + '\n')
        section_end_index += 1
    contents.insert(section_end_index, '\n')

    with open(file_path, 'w') as file:
        file.writelines(contents)

if __name__ == "__main__":
        logging.basicConfig(filename='example.log', filemode='w', encoding='utf-8', level=logging.INFO)
        parser = argparse.ArgumentParser(description='Active Learning Toolset for NLP tasks')

        
        parser.add_argument('--label', type=str, required=True,
                            help='Label of the target class')
        
        parser.add_argument('--label_list', nargs='+', required=True,
                            help='List of all possible labels')
        
        parser.add_argument('--path', type=str, default=PATH_TO_BRAT,
                            help=f'Path to brat directory (default: {PATH_TO_BRAT})')
        
        parser.add_argument('--collection', type=str, default=COLLECTION_NAME,
                            help=f'Name of the annotation collection (default: {COLLECTION_NAME})')
        
        parser.add_argument('--folder', type=str, default=FOLDER_NAME,
                            help=f'Name of the destination folder (default: {FOLDER_NAME})')
        
        parser.add_argument('--num_suggestions', type=int, default=3,
                            help='Number of suggestions to be made by the Active Learning functionality (default: 3)')
        
        parser.add_argument('--token_aggregation', type=str, default='max',
                            help='Strategy that is used for token probability aggregation (default: max)')
        
        parser.add_argument('--iterations', type=int, default='10',
                            help='Number of active learning iterations (default: 10)')
        
        parser.add_argument('--certainty_threshold', type=float, default='90',
                            help='Probability threshold for considering a prediction as certain and adding the annotation to the annotation files (default: 0.9)')
        
        args = parser.parse_args()

        system = System()
        system.set_constant_value(constant_name="TARGET_CLASS", value=args.label)
        system.set_constant_value(constant_name="CERTAINTY_THRESHOLD", value=args.certainty_threshold)

        add_to_config(file_path="config/annotation.conf", type="entities", entities=[SUGGESTION_ANNOTATION_TYPE, args.label, "no-" + args.label])
        add_to_config(file_path="config/visual.conf", type="labels", entities=[SUGGESTION_ANNOTATION_TYPE + " | Annotation suggestion | TBA", args.label + " | " + args.label + " name | " + args.label[:2], "no-" + args.label + " | " + "no-" + args.label + " | no" + args.label[0]])
        add_to_config(file_path="config/visual.conf", type="drawing", entities=["SPAN_DEFAULT	fgColor:black, bgColor:lightgreen, borderColor:darken", "ARC_DEFAULT	color:black, dashArray:-, arrowHead:triangle-5, labelArrow:none", SUGGESTION_ANNOTATION_TYPE + "	bgColor:lightsalmon"])

        system.start_docker()
        
        path_to_collection, file_names = system.get_file_names_from_path(path_to_brat=args.path, folder_name=args.folder, collection_name=args.collection)
        texts = []
        for file_name in file_names:
            if ".txt" in file_name:
                texts.append(TextFile(file_name=file_name, path=path_to_collection).read())

        classifier = SciBertClassifier(num_classes=args.num_suggestions, label=args.label, label_list=args.label_list, token_aggregation=args.token_aggregation)
        _, file_names = system.get_file_names_from_path(path_to_brat="./model", folder_name=None)
        if file_names:
            classifier.load()
        url = "http://localhost:8001/index.xhtml#/"
        url += args.collection + "/" if args.collection else ""
        url += args.folder + "/" if args.folder else ""
        webbrowser.open(url)

        active_learner = ActiveLearning()
        for i in range(args.iterations):
            active_learner.iteration(classifier=classifier, unlabeled_data=texts[:1], num_to_annotate=args.num_suggestions)

        gui = GUI()
        gui.show_custom_popup("End of training", "The Active Learning process has been finished.\n\nYou can now close the window. The model trained on the latest annotations is available under /model/SciBertClassifier.joblib.")

        system.terminate_docker()
