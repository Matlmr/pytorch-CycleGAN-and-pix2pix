import os.path
from data.base_dataset import BaseDataset, get_transform, get_transform_mask
from data.image_folder import make_dataset
from PIL import Image
import random
from numpy import arange
from torchvision.transforms import ToPILImage, ToTensor

# Setting parameters
relax_crop = 50

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        #self.dir_gtA = os.path.join(opt.dataroot, 'gt' + 'A' + 'hair')
        self.dir_gtA = os.path.join(opt.dataroot, 'gt' + '2')
        self.dir_gtB = os.path.join(opt.dataroot, 'gt' + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.gtA_paths = make_dataset(self.dir_gtA)
        self.gtB_paths = make_dataset(self.dir_gtB)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.gtA_paths = sorted(self.gtA_paths)
        self.gtB_paths = sorted(self.gtB_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.gtA_size = len(self.gtA_paths)
        self.gtB_size = len(self.gtB_paths)
        if opt.crop_mask:
            assert(self.A_size == self.gtA_size)
            assert(self.B_size == self.gtB_size)
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        gtA_path = self.gtA_paths[index % self.gtA_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        gtB_path = self.gtB_paths[index_B]
        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        gtA_img = Image.open(gtA_path).convert('L')
        gtB_img = Image.open(gtB_path).convert('L')

        #gtA_img.show()
        #gtB_img.show()

        colors = arange(256) * 0 + 255
        colors[0] = colors[1] = colors[2] = 0
        #gtA_img = gtA_img.point(colors)
        gtB_img = gtB_img.point(colors)

        #gtA_img.show()
        #gtB_img.show()

        #gtB_img = gtB_img.convert('1')

        #gtB_img.show()

        if self.opt.crop_mask:
            
            bboxA = gtA_img.getbbox()
            widthA, heightA = gtA_img.size
            leftA = bboxA[0]-relax_crop
            topA = bboxA[1]-relax_crop
            rightA = bboxA[2]+relax_crop
            botA = bboxA[3]+relax_crop
            A_img = A_img.crop([max(0,leftA), max(0,topA),
                       min(widthA,rightA), min(heightA,botA)])
            gtA_img = gtA_img.crop([max(0,leftA), max(0,topA),
                       min(widthA,rightA), min(heightA,botA)])

            # Zero padding in case the relevant part is near an edge
            if leftA < 0 or topA < 0 or rightA > widthA or botA > heightA:
                padded_img1 = Image.new('RGB', (rightA-leftA, botA-topA), (0,0,0))
                padded_img2 = Image.new('L', (rightA-leftA, botA-topA), (0))
                padded_img1.paste(A_img, (-min(0,leftA), -min(0,topA)))
                padded_img2.paste(gtA_img, (-min(0,leftA), -min(0,topA)))
                A_img = padded_img1
                gtA_img = padded_img2

            bboxB = gtB_img.getbbox()
            widthB, heightB = gtB_img.size
            leftB = bboxB[0]-relax_crop
            topB = bboxB[1]-relax_crop
            rightB = bboxB[2]+relax_crop
            botB = bboxB[3]+relax_crop
            B_img = B_img.crop([max(0,leftB), max(0,topB),
                       min(widthB,rightB), min(heightB,botB)])
            gtB_img = gtB_img.crop([max(0,leftB), max(0,topB),
                       min(widthB,rightB), min(heightB,botB)])

            # Zero padding in case the relevant part is near an edge
            if leftB < 0 or topB < 0 or rightB > widthB or botB > heightB:
                padded_img1 = Image.new('RGB', (rightB-leftB, botB-topB), (0,0,0))
                padded_img2 = Image.new('L', (rightB-leftB, botB-topB), (0))
                padded_img1.paste(B_img, (-min(0,leftB), -min(0,topB)))
                padded_img2.paste(gtB_img, (-min(0,leftB), -min(0,topB)))
                B_img = padded_img1
                gtB_img =padded_img2

        transform_mask = get_transform_mask(self.opt,A_img.size)
        A = transform_mask(A_img)
        transform_mask = get_transform_mask(self.opt,B_img.size)
        B = transform_mask(B_img)
        transform_mask = get_transform_mask(self.opt,gtA_img.size)
        gtA = transform_mask(gtA_img)
        transform_mask = get_transform_mask(self.opt,gtB_img.size)
        gtB = transform_mask(gtB_img)
        
        #A = self.transform(A_img)
        #B = self.transform(B_img)
        #gtA = self.transform(gtA_img)
        #gtB = self.transform(gtB_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        #if input_nc != 4:
        #    return {'A': A, 'B': B,
        #        'A_paths': A_path, 'B_paths': B_path}
        #else:
        return {'A': A, 'B': B,
            'A_paths': A_path, 'B_paths': B_path,
            'gtA': gtA, 'gtB': gtB}



    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'
