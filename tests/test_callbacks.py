from kerashistoryplot import callbacks


def test_api_compatible():
    batch_end_logs = {"this": "is a log"}

    new = callbacks.PlotHistory()
    new.on_train_begin()
    new.on_epoch_begin(epoch=0)
    new.on_train_batch_end(batch=0, logs=batch_end_logs)

    old = callbacks.PlotHistory()
    old.on_train_begin()
    old.on_epoch_begin(epoch=0)
    old.on_batch_end(batch=0, logs=batch_end_logs)

    assert old.history == new.history
