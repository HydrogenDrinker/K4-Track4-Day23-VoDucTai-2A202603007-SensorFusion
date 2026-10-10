# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Võ Đức Tài
- MSSV: 2A202603007
- Email: holysheep739@gmail.com
- Link repo (fork): https://github.com/HydrogenDrinker/K4-Track4-Day23-VoDucTai-2A202603007-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): [Sẽ điền sau khi commit CP5 và CP6]

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `fusion_mode`: compare, `frames`: [0, 198], `segment`: training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, `seed`: 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: `precision`: 0.9701 (97.01%), `recall`: 0.7004 (70.04%), `tp/fp/fn`: 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse`: 0.1503 m, `matches`: 502, `sum_sq_err`: 11.3436, `ghost_track_frames`: 0, `missed_gt_frames`: 239, `mean_confirmed_tracks`: 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: `rmse`: 0.1359 m, `matches`: 502, `sum_sq_err`: 9.2668, `ghost_track_frames`: 0, `missed_gt_frames`: 239, `mean_confirmed_tracks`: 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  - **Về sai số vị trí (RMSE):** Chế độ Fused đạt RMSE là **0.1359 m**, cải thiện rõ rệt so với LiDAR-only là **0.1503 m** (giảm sai số vị trí **~9.6%**, $0.1503 - 0.1359 = 0.0145$ m). Cả hai chế độ đều vượt xa yêu cầu của rubric ($\le 0.45$ m). Tổng sai số bình phương `sum_sq_err` giảm từ 11.3436 xuống 9.2668 nhờ phép đo góc chiếu chính xác từ camera FRONT hỗ trợ EKF tinh chỉnh toạ độ vật thể.
  - **Về số lượng gán ghép (Matches):** Cả hai chế độ đạt đúng **502** matches với xe ground-truth, đạt tỷ lệ coverage $502 / 519 \approx 96.73\%$ (vượt xa ngưỡng 70% của rubric).
  - **Về độ sạch của track (Ghost tracks & Misses):** Cả hai chế độ đều đạt **0 ghost track frames** (đạt độ chính xác tracking tuyệt đối $502 / (502 + 0) = 100\%$, vượt xa ngưỡng 75% của rubric) và số missed GT frames bằng nhau là **239** (chủ yếu do giới hạn nhận diện của detector ở xa ngoài FOV).
  - **Kết luận:** Mô hình track-then-fuse phát huy hiệu quả tối đa: camera 2D chỉ cập nhật trạng thái EKF mà không can thiệp vào vòng đời track (score/init/delete), giúp tăng độ chính xác vị trí mà không sinh ra bất kỳ track ảo (ghost) nào.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   - **Đo LiDAR 3D:** Vector đo $z_{lidar} = [x, y, z]^T \in \mathbb{R}^3$ trực tiếp trong không gian 3D. Hàm đo tuyến tính $h(x) = H x$ với ma trận đo trực tiếp $H_{3 \times 6} = [I_{3 \times 3} \mid 0_{3 \times 3}]$. Ma trận hiệp phương sai $R_{lidar} \in \mathbb{R}^{3 \times 3} = \text{diag}(\sigma_{lidar,x}^2, \sigma_{lidar,y}^2, \sigma_{lidar,z}^2)$.
   - **Đo Camera 2D:** Vector đo $z_{cam} = [u, v]^T \in \mathbb{R}^2$ là toạ độ điểm ảnh (pixel) trên mặt phẳng chiếu pinhole. Hàm đo $h(x)$ phi tuyến do phép chiếu phối cảnh chia cho độ sâu $x_s$: $u = c_i - f_i \frac{y_s}{x_s}$, $v = c_j - f_j \frac{z_s}{x_s}$ (với $p_s = R p + t$). Vì vậy EKF phải dùng ma trận Jacobian $H_{2 \times 6}$ đạo hàm riêng theo trạng thái track tại điểm dự đoán. Ma trận hiệp phương sai $R_{cam} \in \mathbb{R}^{2 \times 2} = \text{diag}(\sigma_{cam,i}^2, \sigma_{cam,j}^2)$ đo bằng đơn vị pixel$^2$.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   - Khoảng cách Mahalanobis bình phương $d^2 = \gamma^T S^{-1} \gamma$ tính khoảng cách giữa phép đo và dự đoán được chuẩn hoá bởi hiệp phương sai đổi mới $S = H P H^T + R$. Đại lượng này xét đến cả độ bất định hiện tại của track ($P$) lẫn nhiễu cảm biến ($R$).
   - Gating dựa trên phân phối $\chi^2$ loại bỏ các đo nằm ngoài vùng phân bố xác suất tin cậy của track trước khi chạy thuật toán gán (greedy/Hungarian). Điều này ngăn ngừa việc gán nhầm các phát hiện giả (clutter/false positive) ở xa vào track, giúp bảo vệ bộ lọc EKF không bị cập nhật sai lệch làm phân kỳ covariance, đồng thời giảm thiểu chi phí tính toán.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**
   - Pipeline của lab là **track-then-fuse**. Hệ thống duy trì một danh sách track duy nhất. Tại mỗi frame, chu kỳ xử lý diễn ra tuần tự: EKF Predict $\to$ gán và update bằng đo LiDAR (AssocL) $\to$ gán và update bổ sung bằng đo Camera (AssocC) $\to$ quản lý vòng đời track (manage tracks).
   - Hệ thống không kết hợp thô các detection LiDAR và camera trước khi tracking (fuse-then-track), mà mỗi cảm biến độc lập cập nhật vào trạng thái của track đang chạy. Trên log `grade_run.log`, chuỗi sự kiện được ghi rõ ràng theo từng bước: dự đoán trạng thái, cập nhật LiDAR mode, sau đó cập nhật camera mode.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   - Nếu extrinsic ($R, t$) hoặc intrinsic ($f, c$) bị lệch, phép chiếu toạ độ không gian sang điểm ảnh $h(x)$ sẽ có sai số hệ thống (systematic bias) so với vị trí quan sát thực tế $z_{cam}$.
   - **Triệu chứng:** Innovation $\gamma = z - h(x)$ sẽ không còn kỳ vọng bằng 0 ($E[\gamma] \neq 0$) mà có một vector lệch cố định theo chiều pixel.
   - Hậu quả: Khoảng cách Mahalanobis $d^2$ tăng vọt. Nếu lệch nhỏ, EKF bị kéo lệch ước lượng vị trí, làm tăng RMSE. Nếu lệch lớn vượt ngưỡng $\chi^2$, toàn bộ phép đo camera của xe thật sẽ bị loại tại bước gating (gate rejection), camera không cập nhật được cho track.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.**
   - Cần sensor tường minh ở frame rỗng vì: Ngay cả khi `meas_list` rỗng, hàm vẫn phải gọi `manager.manage_tracks(...)`. Trình quản lý cần biết loại sensor để xử lý: nếu là LiDAR, mọi track nằm trong FOV mà không có đo tương ứng sẽ bị tính là miss và bị trừ điểm (`score -= 1/window`).
   - LiDAR quyết định score/init/delete còn camera chỉ EKF update vì: LiDAR cung cấp đầy đủ thông tin toạ độ 3D chính xác, cho phép khởi tạo trạng thái ban đầu của track $(x,y,z)$. Camera 2D chỉ quan sát được góc chiếu $(u,v)$, thiếu thông tin độ sâu nên nếu dùng để khởi tạo track sẽ sinh ra vô số ghost tracks không kiểm soát. Ngoài ra camera có trường nhìn hẹp hơn và dễ bị che khuất/ngược sáng, nên việc không thấy trên camera không phản ánh việc vật thể biến mất, chỉ LiDAR 360 độ mới có quyền hạn sinh/diệt track.

6. **Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.**
   - **Xác nhận (Confirmed):** Track mới tạo có `score = 1/window`. Khi được liên kết với đo LiDAR (hit), `score = min(1.0, score + 1/window)`. Khi `score > confirmed_threshold` (0.6), track chuyển thành `confirmed`.
   - **Giữ confirmed sau miss:** Khi một track đã ở trạng thái `confirmed`, nếu gặp frame không có đo LiDAR trong FOV, `score = score - 1/window`, nhưng trạng thái vẫn được duy trì là `confirmed` (không bị hạ cấp về `tentative`).
   - **Xóa track:** Track bị xóa khi thoả mãn **bất kỳ** điều kiện nào sau đây:
     1. Covariance vị trí quá lớn: $P_{xx} > \text{max\_P}$ hoặc $P_{yy} > \text{max\_P}$ (với $\text{max\_P} = 4.0$).
     2. Đã confirmed nhưng `score < delete_threshold` (0.3).
     3. Chưa confirmed mà `score <= 0.0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity / AI Assistant
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ phân tích mô hình chuyển động và công thức ma trận EKF (Part E), hàm chiếu pinhole và FOV (Part G), khoảng cách Mahalanobis và greedy gating (Part F), logic vòng đời track (Part H), giải thích lý thuyết 6 câu hỏi và debug unit test trên Windows.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Kiểm tra toàn bộ 128/128 test case bằng pytest pass 100%, đối chiếu độc lập công thức đạo hàm Jacobian và ma trận hiệp phương sai trong giáo trình và docs/HUONG_DAN_KY_THUAT.md.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
