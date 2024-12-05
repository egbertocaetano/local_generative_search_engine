"""TODO:"""

import os

class Folder:

    """
    TODO:
    """

    def get_files(self, folder_path:str):

        """
        TODO:
        """
        # file_list = []

        # for folder_object in listdir(folder_path):

        #     folder_object_path = join(folder_path, folder_object)

        #     if isfile(folder_object_path):

        #         file_list.append(folder_object_path)

        #     elif isdir(folder_object):

        #         file_list = file_list + self.get_files(folder_path=folder_object)

        # return file_list

        file_list = []

        for dir, _, filenames in os.walk(folder_path):
            for f in filenames:
                file_list.append(os.path.join(dir, f))

        return file_list