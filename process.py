import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
from fer import FER
from fer.utils import draw_annotations
from collections import Counter
import time
import csv
import logging
import os
import re
from pathlib import Path
from typing import Optional, Union
from zipfile import ZipFile

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import streamlit as st
import mimetypes
import plotly.graph_objects as go

# - - - - - - For image input - - - - - - - - - 

def image (pil_img):
    cv_img = cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)
    detector = FER(mtcnn=True)
    faces = detector.detect_emotions(cv_img)
    cv_img = draw_annotations(cv_img, faces)
    pil_img = Image.fromarray(cv.cvtColor(cv_img, cv.COLOR_BGR2RGB))
    return faces, pil_img

def modal_emo(d):
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))]

def emo_freqtable(faces):
    emo_cnt = {'angry': 0,
    'disgust': 0,
    'fear': 0,
    'happy': 0,
    'neutral': 0,
    'sad': 0,
    'surprise': 0}

    cnt_per_frame = Counter([modal_emo(face["emotions"]) for face in faces])

    for emo in cnt_per_frame.keys():
        emo_cnt[emo] += cnt_per_frame[emo]

    return emo_cnt

def img_stats(emo_dict):
    return pd.DataFrame.from_dict({'Emotion': emo_dict.keys(), 'Frequency': emo_dict.values()})


# - - - - - - For video input - - - - - - - - - 

log = logging.getLogger("fer")


class Video2(object):
    def __init__(
        self,
        video_file: str,
        video_name: str,
        outdir: str = "emo_video_output", # name of the output directory storing the images n video
        first_face_only: bool = True,
        tempfile: Optional[str] = None,
    ):
        """Video class for extracting and saving frames for emotion detection.
        :param video_file - str
        :param outdir - str
        :param first_face_only - bool
        :param tempfile - str
        """
        
        # assert os.path.exists(video_file), "Video file not found at {}".format(
        #     os.path.abspath(video_file)
        # )
        
        self.cap = cv.VideoCapture(video_file)     # cap holds the opencv video frame list
        
        # Set the output directory to store the resulting images & video
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        self.outdir = outdir
        
        # Take the first face bounding box over the many neighboring overlapping bounding boxes
        if not first_face_only:
            log.error("Only single-face charting is implemented")
        self.first_face_only = first_face_only
        
        self.tempfile = tempfile
        
        # # Set the filepath of the video & parse the video file name from it
        self.filepath = video_file
        # self.filename = "".join(self.filepath.split("/")[-1])   # The video file name is the last name in the list of names in the filepath (separated by /)
        self.filename = video_name

    @staticmethod
    def _to_dict(data: Union[dict, list]) -> dict:
        emotions = []

        frame = data[0]
        if isinstance(frame, list): # if the frame object is a list, then convert it to a dict
            try:
                emotions = frame[0]["emotions"].keys()
            except IndexError:
                raise Exception("No data in 'data'")
        elif isinstance(frame, dict):
            return data

        dictlist = []

        for data_idx, frame in enumerate(data):
            rowdict = {}
            for idx, face in enumerate(list(frame)):
                if not isinstance(face, dict):
                    break
                rowdict.update({"box" + str(idx): face["box"]})
                rowdict.update(
                    {emo + str(idx): face["emotions"][emo] for emo in emotions}
                )
            dictlist.append(rowdict)
        return dictlist

    def to_pandas(self, data: Union[pd.DataFrame, list]) -> pd.DataFrame:
        """Convert results to pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data

        if not len(data):
            return pd.DataFrame()
        datalist = self._to_dict(data)
        df = pd.DataFrame(datalist)
        if self.first_face_only:
            df = self.get_first_face(df)
        return df

    @staticmethod
    def get_first_face(df: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame), "Must be a pandas DataFrame"
        try:
            int(df.columns[0][-1])
        except ValueError:
            # Already only one face in df
            return df

        columns = [x for x in df.columns if x[-1] == "0"]
        new_columns = [x[:-1] for x in columns]
        single_df = df[columns]
        single_df.columns = new_columns
        return single_df

    def to_csv(self, data, filename="data.csv"):
        """Save data to csv"""

        def key(item):
            key_pat = re.compile(r"^(\D+)(\d+)$")
            m = key_pat.match(item)
            return m.group(1), int(m.group(2))

        dictlist = self._to_dict(data)
        columns = set().union(*(d.keys() for d in dictlist))
        columns = sorted(columns, key=key)  # sort by trailing number (faces)

        with open("data.csv", "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(dictlist)
        return dictlist

    def _increment_frames(self, frame, faces, video_id, root):
        # Save images to `self.outdir`
        imgpath = os.path.join(
            self.outdir, (video_id or root) + str(self.frameCount) + ".jpg" # this is where the image is named!!!!! (We only do this for saving the image bcos the many frames need a unique name, the video no need)
        )

        if self.annotate_frames:
            frame = draw_annotations(frame, faces, boxes=True, scores=True)

        if self.save_frames:
            cv.imwrite(imgpath, frame)    # save the annotated output frame in the image path created at the beginning

        if self.display:
            cv.imshow("Video", frame)

        if self.save_video:
            self.videowriter.write(frame)   # save the annotated output frame into the output video    

        self.frameCount += 1

    # Create a cv.VideoWriter object (store the output video) to later write the output frames into this in the video processing loop
    def _save_video(self, outfile: str, fps: int, width: int, height: int):
        if os.path.isfile(outfile):
            os.remove(outfile)
            log.info("Deleted pre-existing {}".format(outfile))
        if self.tempfile and os.path.isfile(self.tempfile):
            os.remove(self.tempfile)
        fourcc = cv.VideoWriter_fourcc("m", "p", "4", "v")
        videowriter = cv.VideoWriter(
            self.tempfile or outfile, fourcc, fps, (width, height), True
        )
        return videowriter

    def _close_video(self, outfile, save_frames, zip_images):
        self.cap.release()
        if self.display or self.save_video:
            self.videowriter.release()

        if self.save_video:
            log.info("Completed analysis: saved to {}".format(self.tempfile or outfile))
            if self.tempfile:
                os.replace(self.tempfile, outfile)

        if save_frames and zip_images:
            log.info("Starting to Zip")
            outdir = Path(self.outdir)
            zip_dir = outdir / "images.zip"
            images = sorted(list(outdir.glob("*.jpg")))
            total = len(images)
            i = 0
            with ZipFile(zip_dir, "w") as zip:
                for file in images:
                    zip.write(file, arcname=file.name)
                    os.remove(file)
                    i += 1
                    if i % 50 == 0:
                        log.info(f"Compressing: {i*100 // total}%")
            log.info("Zip has finished")
    
    def __del__(self):
        cv.destroyAllWindows()
    
    def to_format(self, data, format):
        """Return data in format."""
        methods_lookup = {"csv": self.to_csv, "pandas": self.to_pandas} #this is a dict to select the code to use based on the "format" parameter
        return methods_lookup[format](data)

    def modal_key(self, d):
        """Return the modal key in a dict."""
        v = list(d.values())
        k = list(d.keys())
        return k[v.index(max(v))]

    def emo_freq_row(self, faces, frame_count):
        """Return a single row dataframe of the freq table of each emotion in a frame."""
        emo_cnt_dict = {
            'time':     [0],
            'angry':    [0],
            'disgust':  [0],
            'fear':     [0],
            'happy':    [0],
            'neutral':  [0],
            'sad':      [0],
            'surprise': [0]
        }

        emo_cnt_dict['time'][0] = frame_count - 1

        cnt_per_frame = Counter([self.modal_key(face["emotions"]) for face in faces])

        for emo in cnt_per_frame.keys():
            emo_cnt_dict[emo][0] += cnt_per_frame[emo]

        return pd.DataFrame(emo_cnt_dict)

    def analyze(
        self,
        detector,  # fer.FER instance
        display: bool = False,
        output: str = "csv",
        frequency: Optional[int] = None,
        max_results: int = None,
        video_id: Optional[str] = None,
        save_frames: bool = True,
        save_video: bool = True,
        annotate_frames: bool = True,
        zip_images: bool = True
    ) -> list:
        """Recognize facial expressions in video using `detector`.
        Args:
            detector (fer.FER): facial expression recognizer
            display (bool): show images with cv.imshow
            output (str): csv or pandas
            frequency (int): inference on every nth frame (higher number is faster)
            max_results (int): number of frames to run inference before stopping
            video_id (str): filename for saving
            save_frames (bool): saves frames to directory
            save_video (bool): saves output video
            annotate_frames (bool): add emotion labels
            zip_images (bool): compress output
        Returns:
            data (DataFrame): dataframe of the emotions at each second of the video
        """

        emo_freq_df = pd.DataFrame()

        if frequency is None:
            frequency = 1   # by default, it goes thru every frame
        else:
            frequency = int(frequency)

        self.display = display  # Pass the function params to the video2 oject properties    
        self.save_frames = save_frames
        self.save_video = save_video
        self.annotate_frames = annotate_frames

        results_nr = 0

        # Open video
        assert self.cap.open(self.filepath), "Video capture not opening"
        self.__emotions = detector._get_labels().items()    # get the emotion labels from the FER object
        
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        pos_frames = self.cap.get(cv.CAP_PROP_POS_FRAMES)
        assert int(pos_frames) == 0, "Video not at index 0"

        self.frameCount = 0     # Initialise frame count with 0
        selectedFrameCounts = 0 # Equivalent to the selected frames count

        height, width = (
            int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)),   # get these parameters from the opencv video capture object
            int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)),
        )

        fps = self.cap.get(cv.CAP_PROP_FPS)    # get these parameters from the opencv video capture object
        length = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))    # total frame count
        assert fps and length, "File {} not loaded".format(self.filepath)
        # actual_frame_count = int(-(-length // fps))  # round up the selected frame count
        actual_frame_count = int(length // fps)
        log.info(
            "{:.2f} fps, {:.2f} seconds, {} selected frames / {} total frames".format(fps, length / fps, actual_frame_count, length)
        )

        if self.save_frames:
            os.makedirs(self.outdir, exist_ok=True)     # make the output folder (ignore if it exists already)
            log.info(f"Making directories at {self.outdir}")
        root, ext = os.path.splitext(self.filename)   # Get the file name and split it to its name + file extension name
        outfile = os.path.join(self.outdir, f"video{ext}")  # move the file to the output folder by appending the output folder path to the extracted filename. We also renamed the filename by suffixing with file_output.ext. We use the root file_output later in increment_frames() to name the output image

        if save_video:  # Prepare the VideoWriter object to save the output frames into
            self.videowriter = self._save_video(outfile, fps, width, height)

        with logging_redirect_tqdm():
            pbar = tqdm(total=actual_frame_count, unit="frames")    # the progress display of the video processing progress. Here we create a progress bar (pbar) object with length of the video total frames count, and unit of frames
        st_pbar = st.progress(0)
        curr_prog = 0
        video_processing = st.empty()


        while self.cap.isOpened():  # main detection on the frames IS HERE. Video processing loop frame-by-frame
            ret, frame = self.cap.read()
            if not ret:  # end of video
                break

            if frame is None:
                log.warn("Empty frame")
                continue
            
            # Selects one frame from each video second
            if self.frameCount % frequency != 0:    # Loop thru video frames at the set frequency (Genius!)
                # print(round(self.frameCount / frequency, 2))
                self.frameCount += 1
                continue
            
            # Beyond this part of the code, the frame is one of the selected frame for processing, 
            selectedFrameCounts += 1

            # Get faces and detect emotions; coordinates are for unpadded frame
            try:
                faces = detector.detect_emotions(frame)     # detect emotions!!!!
            except Exception as e:
                log.error(e)
                break

            self._increment_frames(frame, faces, video_id, root)    # annotate the output frame, and name its file

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

            # Changed here
            if faces:   # if detected any face, append to the emo-freq timeline dataframe
                emo_freq_df = pd.concat([emo_freq_df, self.emo_freq_row(faces, selectedFrameCounts)])

            results_nr += 1
            if max_results and results_nr > max_results:    # check if frame count exceeds max frames allowed to process
                break

            pbar.update(1)
            curr_prog = int((100 / actual_frame_count) * (selectedFrameCounts))
            # print(f'Processing {selectedFrameCounts} / {actual_frame_count} frames. But printing progress {curr_prog}')
            st_pbar.progress(curr_prog)
            if curr_prog < 100: video_processing.text(f'Processing {curr_prog}% ...')
            else: video_processing.text(f'Processing completed!')

        pbar.close()
        st_pbar.progress(100)
        video_processing.text(f'Processing completed!')
        self._close_video(outfile, save_frames, zip_images)   # close the video, save the images into a zip files
        return emo_freq_df.reset_index(drop=True)

    def mean_cnt(self, df):
        """Return the average frequency of each emotion in a single row dataframe"""
        emos = list(FER._get_labels().values())
        df = pd.DataFrame({
            'Mean Frequency' : df[emos].mean().astype(int)
        })
        return df



def process_video(video_file, video_name):
    video = Video2(video_file, video_name)
    detector = FER(mtcnn=True)

    start_time = time.time()
    # Output list of dictionaries
    emo_timeline_df = video.analyze(
        detector,  
        display = False,
        frequency = 30,
        video_id = None,
        save_frames = True,
        save_video = True,
        annotate_frames = True,
        zip_images = True,
    )
    end_time = time.time()

    print("\nTime taken = ", round(end_time - start_time, 0), "seconds\n")

    mean_emo_cnt = video.mean_cnt(emo_timeline_df)
    # Plot emotions
    # df.plot()
    # plt.show()

    return emo_timeline_df, mean_emo_cnt

def video_stats(df):
    emos = list(FER._get_labels().values())
    fig = go.Figure()
    for emo in emos:
        fig.add_trace(go.Scatter(x=df['time'], y=df[emo], name=emo))
    fig.update_traces(mode='lines+markers')
    fig.update_layout(title='Timeline of emotions in video',
                    xaxis_title='Video time (seconds)',
                    yaxis_title='Frequency (people)')
    return fig

# - - - - - - Utility - - - - - - - - - 
mimetypes.init()

def mediaFileType(fileName):
    '''
    Returns 0 for images, 1 for videos
    '''
    mimestart = mimetypes.guess_type(fileName)[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]

        if mimestart == 'image':    return 1
        elif mimestart == 'video':  return 2
    
    return False
