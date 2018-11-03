from data_loader.tf_records_singulation_sim import create_tfrecords_from_dir

if __name__ == '__main__':
    create_tfrecords_from_dir("../data/source", "../data/destination", n_sequences_per_batch=10)

