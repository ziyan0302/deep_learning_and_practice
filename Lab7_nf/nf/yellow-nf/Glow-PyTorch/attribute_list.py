attribute_file = '/home/arg/courses/machine_learning/homework/deep_learning_and_practice/Lab7/dataset/task_2/' + 'CelebA-HQ-attribute-anno.txt'
f = open(attribute_file)
lines = f.readlines()
# print(lines[1])
attributes = lines[1].split()
# print(attributes)
for i, a in enumerate(attributes):
    # print(i, a)



    '''
    0 5_o_Clock_Shadow
    1 Arched_Eyebrows
    2 Attractive
    3 Bags_Under_Eyes
    4 Bald
    5 Bangs
    6 Big_Lips
    7 Big_Nose
    8 Black_Hair
    9 Blond_Hair
    10 Blurry
    11 Brown_Hair
    12 Bushy_Eyebrows
    13 Chubby
    14 Double_Chin
    15 Eyeglasses
    16 Goatee
    17 Gray_Hair
    18 Heavy_Makeup
    19 High_Cheekbones
    20 Male
    21 Mouth_Slightly_Open
    22 Mustache
    23 Narrow_Eyes
    24 No_Beard
    25 Oval_Face
    26 Pale_Skin
    27 Pointy_Nose
    28 Receding_Hairline
    29 Rosy_Cheeks
    30 Sideburns
    31 Smiling
    32 Straight_Hair
    33 Wavy_Hair
    34 Wearing_Earrings
    35 Wearing_Hat
    36 Wearing_Lipstick
    37 Wearing_Necklace
    38 Wearing_Necktie
    39 Young
    '''

print(lines[3])
attributes_1 = lines[3].split()
for i , a in enumerate(attributes_1[1:]):
    print(i ,a)

