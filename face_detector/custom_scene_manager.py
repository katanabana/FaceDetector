from scenedetect.scene_manager import *


class CustomSceneManager(SceneManager):
    def custom_detect_scenes(
        self,
        video: VideoStream = None,
        duration: Optional[FrameTimecode] = None,
        end_time: Optional[FrameTimecode] = None,
        frame_skip: int = 0,
        show_progress: bool = False,
        frame_source: Optional[VideoStream] = None,
    ):
        """Perform scene detection on the given video using the added SceneDetectors, returning the
        number of frames processed. Results can be obtained by calling :meth:`get_scene_list` or
        :meth:`get_cut_list`.

        Video decoding is performed in a background thread to allow scene detection and frame
        decoding to happen in parallel. Detection will continue until no more frames are left,
        the specified duration or end time has been reached, or :meth:`stop` was called.

        Arguments:
            video: VideoStream obtained from either `scenedetect.open_video`, or by creating
                one directly (e.g. `scenedetect.backends.opencv.VideoStreamCv2`).
            duration: Amount of time to detect from current video position. Cannot be
                specified if `end_time` is set.
            end_time: Time to stop processing at. Cannot be specified if `duration` is set.
            frame_skip: Not recommended except for extremely high frame rate videos.
                Number of frames to skip (i.e. process every 1 in N+1 frames,
                where N is frame_skip, processing only 1/N+1 percent of the video,
                speeding up the detection time at the expense of accuracy).
                `frame_skip` **must** be 0 (the default) when using a StatsManager.
            show_progress: If True, and the ``tqdm`` module is available, displays
                a progress bar with the progress, frame rate, and expected time to
                complete processing the video frame source.
            frame_source: [DEPRECATED] DO NOT USE. For compatibility with previous version.
        Raises:
            ValueError: `frame_skip` **must** be 0 (the default) if the SceneManager
                was constructed with a StatsManager object.
        """
        if frame_source is not None:
            video = frame_source
        if video is None:
            raise TypeError(
                "detect_scenes() missing 1 required positional argument: 'video'"
            )
        if frame_skip > 0 and self.stats_manager is not None:
            raise ValueError("frame_skip must be 0 when using a StatsManager.")
        if duration is not None and end_time is not None:
            raise ValueError("duration and end_time cannot be set at the same time!")
        if duration is not None and isinstance(duration, (int, float)) and duration < 0:
            raise ValueError("duration must be greater than or equal to 0!")
        if end_time is not None and isinstance(end_time, (int, float)) and end_time < 0:
            raise ValueError("end_time must be greater than or equal to 0!")

        self._base_timecode = video.base_timecode

        if self._stats_manager is not None:
            self._stats_manager._base_timecode = self._base_timecode

        start_frame_num: int = video.frame_number
        if end_time is not None:
            end_time = self._base_timecode + end_time
        elif duration is not None:
            end_time = (self._base_timecode + duration) + start_frame_num

        total_frames = 0
        if video.duration is not None:
            if end_time is not None and end_time < video.duration:
                total_frames = end_time - start_frame_num
            else:
                total_frames = video.duration.get_frames() - start_frame_num

        # Calculate the desired downscale factor and log the effective resolution.
        if self.auto_downscale:
            downscale_factor = compute_downscale_factor(frame_width=video.frame_size[0])
        else:
            downscale_factor = self.downscale
        if downscale_factor > 1:
            logger.info(
                "Downscale factor set to %d, effective resolution: %d x %d",
                downscale_factor,
                video.frame_size[0] // downscale_factor,
                video.frame_size[1] // downscale_factor,
            )

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(
                total=int(total_frames),
                unit="frames",
                desc=PROGRESS_BAR_DESCRIPTION % 0,
                dynamic_ncols=True,
            )

        frame_queue = queue.Queue(MAX_FRAME_QUEUE_LENGTH)
        self._stop.clear()
        decode_thread = threading.Thread(
            target=SceneManager._decode_thread,
            args=(self, video, frame_skip, downscale_factor, end_time, frame_queue),
            daemon=True,
        )
        decode_thread.start()
        frame_im = None

        logger.info("Detecting scenes...")
        while not self._stop.is_set():
            next_frame, position = frame_queue.get()
            if next_frame is None and position is None:
                break
            if next_frame is not None:
                frame_im = next_frame
            new_cuts, end = self.custom_process_frame(position.frame_num, frame_im)
            if end:
                yield end
            if progress_bar is not None:
                if new_cuts:
                    progress_bar.set_description(
                        PROGRESS_BAR_DESCRIPTION % len(self._cutting_list),
                        refresh=False,
                    )
                progress_bar.update(1 + frame_skip)

        if progress_bar is not None:
            progress_bar.set_description(
                PROGRESS_BAR_DESCRIPTION % len(self._cutting_list), refresh=True
            )
            progress_bar.close()
        # Unblock any puts in the decode thread before joining. This can happen if the main
        # processing thread stops before the decode thread.
        while not frame_queue.empty():
            frame_queue.get_nowait()
        decode_thread.join()

        if self._exception_info is not None:
            raise self._exception_info[1].with_traceback(self._exception_info[2])

        self._last_pos = video.position
        self._post_process(video.position.frame_num)

    def custom_process_frame(
        self, frame_num: int, frame_im: np.ndarray
    ) -> [bool, int | None]:
        """Add any cuts detected with the current frame to the cutting list. Returns True if any new
        cuts were detected, False otherwise."""
        new_cuts = False
        # being processed. Allow detectors to specify the max frame lookahead they require
        # (i.e. any event will never be more than N frames behind the current one).
        self._frame_buffer.append(frame_im)
        # frame_buffer[-1] is current frame, -2 is one behind, etc.
        # so index based on cut frame should be [event_frame - (frame_num + 1)]
        self._frame_buffer = self._frame_buffer[-(self._frame_buffer_size + 1) :]
        end = None
        for detector in self._detector_list:
            cuts = detector.process_frame(frame_num, frame_im)
            self._cutting_list += cuts
            new_cuts = True if cuts else False
            if cuts:
                end = cuts[-1]
        for detector in self._sparse_detector_list:
            events = detector.process_frame(frame_num, frame_im)
            self._event_list += events
            if events:
                end = events[-1][0]
        return new_cuts, end
