class image2csv():
    def __init__(self,input_images = "./data/CTC/Training/Fluo-N2DH-SIM+",
    input_masks  = "./data/CTC/Training/Fluo-N2DH-SIM+",
    input_seg    = "./data/CTC/Training/Fluo-N2DH-SIM+",
    input_model  = "",
    output_csv   = "./data/basic_features/" ,
    basic= True,
    sequences=['01', '02'],
    seg_dir='_GT/TRA'):
        self.input_images = input_images
        self.input_masks = input_masks
        self.input_model = input_model
        self.output_csv = output_csv
        self.basic = basic
        self.sequences = sequences
        self.seg_dir = seg_dir
    def create_csv_with_default(self):
        import src.preprocess_seq2graph_2d as sq 
        sq.create_csv(self.input_images, self.input_masks, self.input_seg,
               self.input_model, self.output_csv, self.basic,
               self.sequences, self.seg_dir)