import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
import numpy as np
import pandas as pd


def lossAccuracyPlot():
    # fmt: off
    train_loss = [1.1855643627407417, 0.6895626479869134, 0.4844446550189363, 0.392544977471219, 0.28370740846872133, 0.24960422573178884, 0.2339322637468009, 0.12144147727506753, 0.09985293610434462, 0.08817123128783962, 0.08494996108130008, 0.07935560980016625, 0.0686186473456785, 0.08071641599482648, 0.05487372602886064, 0.0778755714934081, 0.08030256245678263, 0.06320578288387449, 0.06380200589785957, 0.06973850625833738, 0.06332170909346771, 0.0775999084437232, 0.06466697200652453, 0.06140468684967008, 0.07295689427303052, 0.06361360865209398]
    test_loss = [3.8205168793201447, 1.6514771375656128, 1.4812007712051272, 1.406359465740621, 2.356638709947467, 1.9666255502123386, 1.0625124773140997, 1.1853534804745578, 0.9228471004930325, 1.1555449654776604, 1.001512357063126, 1.183635949427262, 1.0318782073315234, 1.2754198391214013, 0.8838223907873034, 1.0681233234480023, 0.8403986908420921, 1.014668609640561, 0.969063121072948, 1.1796127974446864, 0.9598910774346441, 0.91817874775175, 0.9028654954414814, 1.044740099091083, 0.905268820897676, 0.959599149376154]

    train_acc = [0.5586, 0.7623, 0.8327, 0.8680, 0.9047, 0.9204, 0.9312, 0.9660, 0.9730, 0.9753, 0.9737, 0.9807, 0.9789, 0.9782, 0.9872, 0.9800, 0.9795, 0.9836, 0.9849, 0.9836, 0.9856, 0.9780, 0.9840, 0.9854, 0.9829, 0.9816]
    test_acc = [0.3620, 0.5500, 0.6020, 0.6400, 0.4920, 0.6300, 0.6920, 0.6580, 0.7360, 0.6840, 0.6800, 0.6700, 0.7400, 0.6500, 0.7300, 0.6780, 0.7600, 0.6820, 0.7280, 0.7040, 0.7180, 0.7400, 0.7280, 0.7120, 0.7120, 0.6980]
    # fmt: on

    fig, ax = plt.subplots(2, figsize=(7, 10))

    ax[0].plot(train_loss, label="Train")
    ax[0].plot(test_loss, label="Test")
    ax[0].axvline(x=len(train_loss) - 10, color="red", label="Early Stopping")
    ax[0].legend()
    ax[0].set_ylabel("Loss")

    ax[1].plot(train_acc, label="Train")
    ax[1].plot(test_acc, label="Test")
    ax[1].axvline(x=len(train_acc) - 10, color="red", label="Early Stopping")
    ax[1].legend()
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    plt.show()


def accuracyComparisonPlot():
    # fmt: off
    alexnet_accuracy_traditional = [0.5204918032786885, 0.6101010101010101, 0.6310483870967742, 0.6544715447154471, 0.784, 0.7626774847870182, 0.725609756097561, 0.6673387096774194, 0.6707070707070707, 0.75]
    alexnet_accuracy_other = [0.5266393442622951, 0.6282828282828283, 0.6048387096774194, 0.6646341463414634, 0.75, 0.7809330628803245, 0.7479674796747967, 0.6229838709677419, 0.6121212121212121, 0.71]

    resnet18_accuracy_traditional = [0.8155737704918032, 0.8626262626262626, 0.9435483870967742, 0.8922764227642277, 0.96, 0.8762677484787018, 0.9390243902439024, 0.8528225806451613, 0.795959595959596, 0.848]
    resnet18_accuracy_other = [0.5942622950819673, 0.7414141414141414, 0.8931451612903226, 0.8130081300813008, 0.962, 0.8032454361054767, 0.9247967479674797, 0.7741935483870968, 0.6747474747474748, 0.706]

    resnet50_accuracy_traditional = [0.8258196721311475, 0.8828282828282829, 0.9112903225806451, 0.8963414634146342, 0.956, 0.8823529411764706, 0.9776422764227642, 0.8165322580645161, 0.9191919191919192, 0.85]
    resnet50_accuracy_other = [0.6577868852459017, 0.896969696969697, 0.8528225806451613, 0.8577235772357723, 0.93, 0.821501014198783, 0.9065040650406504, 0.8104838709677419, 0.8525252525252526, 0.798]

    vgg_accuracy_traditional = [0.694672131147541, 0.8525252525252526, 0.8487903225806451, 0.806910569105691, 0.918, 0.8640973630831643, 0.8841463414634146, 0.7399193548387096, 0.7515151515151515, 0.838]
    vgg_accuracy_other = [0.6352459016393442, 0.8363636363636363, 0.8044354838709677, 0.8109756097560976, 0.908, 0.8397565922920892, 0.8841463414634146, 0.7459677419354839, 0.7313131313131314, 0.81]
    # fmt: on
    ticks = ["AlexNet", "ResNet18", "ResNet50", "VGG"]

    traditional_accuracies = [
        alexnet_accuracy_traditional,
        resnet18_accuracy_traditional,
        resnet50_accuracy_traditional,
        vgg_accuracy_traditional,
    ]
    other_accuracies = [
        alexnet_accuracy_other,
        resnet18_accuracy_other,
        resnet50_accuracy_other,
        vgg_accuracy_other,
    ]

    def set_box_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)

    fig = plt.figure()

    bpl = plt.boxplot(
        traditional_accuracies,
        positions=np.array(range(len(traditional_accuracies))) * 2.0 - 0.4,
        sym="",
        widths=0.6,
    )
    bpr = plt.boxplot(
        other_accuracies,
        positions=np.array(range(len(other_accuracies))) * 2.0 + 0.4,
        sym="",
        widths=0.6,
    )
    set_box_color(bpl, "#D7191C")  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, "#2C7BB6")

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c="#D7191C", label="Traditional Approach")
    plt.plot([], c="#2C7BB6", label="Our Approach")
    plt.legend(loc="lower right")

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.yticks(np.arange(0.5, 1.05, 0.05))
    plt.xlim(-1.2, len(ticks) * 2 - 0.8)
    plt.ylim(0.5, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model Architecture")
    plt.gca().yaxis.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()


def HeatMapPlot():
    heatmap = np.array(
        [
            [497, 15, 78, 123, 189],
            [24, 804, 14, 76, 111],
            [17, 27, 944, 27, 14],
            [102, 95, 153, 610, 77],
            [38, 37, 28, 39, 808],
        ]
    )
    classes = ["7042", "7051", "7055", "7133", "others"]
    heatmap = np.round(heatmap / np.sum(heatmap, axis=1), 5) * 100
    df_cm = pd.DataFrame(
        heatmap, index=[i for i in classes], columns=[i for i in classes]
    )
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(df_cm, annot=True, fmt="g", cmap=sns.color_palette("Blues", 10))
    for t in ax.texts:
        t.set_text(t.get_text() + "%")
    plt.xlabel("predicted")
    plt.ylabel("truth")
    plt.show()


if __name__ == "__main__":
    HeatMapPlot()
