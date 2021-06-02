# # 2) TO change one certain line:
# for Num_layer_revise in range(1,41):
# # Num_layer_revise = 1

#     data = ''
#     num_line = 0
#     Num_line_revise = 3+Num_layer_revise*2

#     with open('prune_ratio_WNLI_v0.yaml', 'r+') as f:
#         for line in f.readlines():
#             num_line += 1
#             if num_line == Num_line_revise:
#                 line = "        0.5\n"

#             data += line

#     with open("prune_ratio_WNLI_v"+str(int(100+Num_layer_revise))+".yaml", 'w+') as f:
#         f.writelines(data)


# 2) TO change one certain line:
for Num_layer_revise in range(1,41):
# Num_layer_revise = 1

    data = ''
    num_line = 0
    Num_line_revise = 3+Num_layer_revise*2

    with open('prune_ratio_WNLI_v0.yaml', 'r+') as f:
        for line in f.readlines():
            num_line += 1
            if num_line == Num_line_revise:
                line = "        0.5\n"

            data += line

    with open("prune_ratio_WNLI_v"+str(int(100+Num_layer_revise))+".yaml", 'w+') as f:
        f.writelines(data)
