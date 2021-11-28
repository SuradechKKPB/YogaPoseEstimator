def activate_software():

    cv2.namedWindow("RSI Prevention using Yoga Pose")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv2.imshow("RSI Prevention using Yoga Pose", frame)
        rval, frame = vc.read()
        frame = imutils.resize(frame, width=600) 
        key = cv2.waitKey(1)
        if key == 27: # exit on ESC
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(frame.shape)
            plt.imshow(frame)
            cv2.imwrite("test01.jpg", frame)
            break

        pose = False
        if pose:
            cv2.putText(frame, '{}'.format(pose), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Pose Detected', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.destroyWindow("preview")
    vc.release()