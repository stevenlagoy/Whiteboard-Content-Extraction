
import pytest

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.web.domain.errors import ConflictError, NotFoundError, ValidationError
from whiteboard_extraction.web.services.library import NAME_LIMIT

# ---------------------------------------------------------------- folders

def test_folder_names_are_kept_exactly_as_entered(library):
    folder = library.create_folder("  CS_593_deep_learning ")
    assert folder.name == "CS_593_deep_learning"


@pytest.mark.parametrize("name", ["", "   ", None, "x" * (NAME_LIMIT + 1)])
def test_folder_name_validation(library, name):
    with pytest.raises(ValidationError):
        library.create_folder(name)


def test_a_folder_id_can_be_chosen(library):
    assert library.create_folder("Fixed", folder_id="my-id").id == "my-id"


def test_duplicate_folder_names_conflict_case_insensitively(library):
    library.create_folder("Algebra")
    with pytest.raises(ConflictError):
        library.create_folder("algebra")


def test_same_name_is_allowed_under_different_parents(library):
    a = library.create_folder("A")
    b = library.create_folder("B")
    library.create_folder("Week 1", a.id)
    assert library.create_folder("Week 1", b.id).parent_id == b.id


def test_create_folder_under_missing_parent(library):
    with pytest.raises(NotFoundError):
        library.create_folder("X", "nope")


def test_rename_folder(library):
    folder = library.create_folder("Old")
    assert library.rename_folder(folder.id, "New_name").name == "New_name"
    assert library.get_folder(folder.id).name == "New_name"


def test_rename_folder_to_its_own_name_is_allowed(library):
    folder = library.create_folder("Same")
    assert library.rename_folder(folder.id, "Same").name == "Same"


def test_rename_folder_conflicts_with_sibling(library):
    library.create_folder("One")
    two = library.create_folder("Two")
    with pytest.raises(ConflictError):
        library.rename_folder(two.id, "one")


def test_delete_empty_folder(library):
    folder = library.create_folder("Empty")
    assert library.delete_folder(folder.id).id == folder.id
    with pytest.raises(NotFoundError):
        library.get_folder(folder.id)


def test_delete_folder_with_video_or_subfolder_conflicts(library, make_video):
    with_video = library.create_folder("V")
    make_video(folder_id=with_video.id)
    with_child = library.create_folder("C")
    library.create_folder("Child", with_child.id)
    for folder in (with_video, with_child):
        with pytest.raises(ConflictError):
            library.delete_folder(folder.id)


def test_folder_path_and_label(library):
    a = library.create_folder("CS 593")
    b = library.create_folder("Week 1", a.id)
    assert [f.name for f in library.folder_path(b.id)] == ["CS 593", "Week 1"]
    assert library.folder_label(b.id) == "CS 593 › Week 1"
    assert library.folder_path(None) == []
    assert library.folder_label(None) == "My files"


def test_folder_choices_are_in_tree_order_with_depth(library):
    b = library.create_folder("beta")
    a = library.create_folder("Alpha")
    library.create_folder("Child", a.id)
    labels = [(c.label, c.depth) for c in library.folder_choices()]
    assert labels == [("My files", 0), ("Alpha", 1), ("Alpha › Child", 2), ("beta", 1)]
    assert library.folder_choices()[0].id is None and b.id in [c.id for c in library.folder_choices()]


def test_folder_tree_sorts_folders_by_name_and_videos_by_position(library, make_video):
    z = library.create_folder("Zed")
    a = library.create_folder("alpha")
    first = make_video("first", folder_id=a.id)
    second = make_video("second", folder_id=a.id)
    loose = make_video("loose")
    tree = library.folder_tree()
    assert [n.folder.name for n in tree.folders] == ["alpha", "Zed"]
    assert [v.id for v in tree.folders[0].videos] == [first.id, second.id]
    assert [v.id for v in tree.videos] == [loose.id]
    assert tree.folders[1].folder.id == z.id and tree.folders[1].videos == ()


# ----------------------------------------------------------------- videos

def test_add_video_appends_in_its_folder(library, make_video):
    folder = library.create_folder("F")
    a = make_video("a", folder_id=folder.id)
    b = make_video("b", folder_id=folder.id)
    assert (a.position, b.position) == (0, 1)
    assert make_video("elsewhere").position == 0


def test_add_video_rejects_unknown_folder_and_bad_names(library):
    with pytest.raises(NotFoundError):
        library.add_video(name="x", folder_id="nope", length_seconds=10)
    with pytest.raises(ValidationError):
        library.add_video(name=" ", folder_id=None, length_seconds=10)


@pytest.mark.parametrize(
    "length, start, end",
    [(0, 0, None), (100, -1, None), (100, 100, None), (100, 50, 50), (100, 50, 40), (100, 0, 120)],
)
def test_add_video_trim_validation(library, length, start, end):
    with pytest.raises(ValidationError):
        library.add_video(name="x", folder_id=None, length_seconds=length, trim_start=start, trim_end=end)


def test_trim_properties(make_video):
    video = make_video(length=100, trim_start=10, trim_end=70)
    assert (video.clip_end, video.clip_seconds, video.is_trimmed) == (70, 60, True)
    whole = make_video(length=100)
    assert (whole.clip_end, whole.clip_seconds, whole.is_trimmed) == (100, 100, False)


def test_rename_video_sets_edited_at(library, make_video, clock):
    video = make_video("Old")
    clock.advance(minutes=5)
    renamed = library.rename_video(video.id, " New ")
    assert renamed.name == "New" and renamed.edited_at == clock.now
    with pytest.raises(ValidationError):
        library.rename_video(video.id, "")


def test_favorites_are_alphabetical(library, make_video):
    b = make_video("b video")
    a = make_video("A video")
    make_video("plain")
    library.set_favorite(b.id, True)
    library.set_favorite(a.id, True)
    assert [v.name for v in library.favorites()] == ["A video", "b video"]
    library.set_favorite(a.id, False)
    assert [v.name for v in library.favorites()] == ["b video"]


def test_recent_orders_by_latest_activity_and_limits(library, make_video, clock):
    old = make_video("old")
    clock.advance(hours=1)
    new = make_video("new")
    clock.advance(hours=1)
    library.rename_video(old.id, "old edited")
    assert [v.name for v in library.recent(10)] == ["old edited", "new"]
    assert [v.id for v in library.recent(1)] == [old.id]
    assert new.id in [v.id for v in library.recent(2)]


class TestMoveAndOrder:
    @pytest.fixture
    def setup(self, library, make_video):
        f1, f2 = library.create_folder("F1"), library.create_folder("F2")
        a, b, c = (make_video(n, folder_id=f1.id) for n in "abc")
        x = make_video("x", folder_id=f2.id)
        return library, f1, f2, a, b, c, x

    @staticmethod
    def names(library, folder_id):
        return [v.name for v in library.videos_in(folder_id)]

    def test_move_to_another_folder_appends(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.move_video(a.id, f2.id)
        assert self.names(library, f2.id) == ["x", "a"]
        assert self.names(library, f1.id) == ["b", "c"]

    def test_move_before_a_sibling(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.move_video(c.id, f1.id, before_id=a.id)
        assert self.names(library, f1.id) == ["c", "a", "b"]

    def test_move_before_in_another_folder(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.move_video(b.id, f2.id, before_id=x.id)
        assert self.names(library, f2.id) == ["b", "x"]
        assert self.names(library, f1.id) == ["a", "c"]

    def test_move_to_top_level_and_back(self, setup):
        library, f1, f2, a, *_ = setup
        assert library.move_video(a.id, None).folder_id is None
        assert self.names(library, None) == ["a"]

    def test_move_before_itself_changes_nothing(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.move_video(b.id, f1.id, before_id=b.id)
        assert self.names(library, f1.id) == ["a", "b", "c"]

    def test_move_rejects_bad_targets(self, setup):
        library, f1, f2, a, b, c, x = setup
        with pytest.raises(NotFoundError):
            library.move_video(a.id, "nope")
        with pytest.raises(NotFoundError):
            library.move_video("nope", f1.id)
        with pytest.raises(ValidationError):
            library.move_video(a.id, f2.id, before_id=b.id)  # b is not in f2

    def test_shift_up_and_down(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.shift_video(c.id, "up")
        assert self.names(library, f1.id) == ["a", "c", "b"]
        library.shift_video(a.id, "down")
        assert self.names(library, f1.id) == ["c", "a", "b"]

    def test_shift_at_the_edges_is_a_no_op(self, setup):
        library, f1, f2, a, b, c, x = setup
        library.shift_video(a.id, "up")
        library.shift_video(c.id, "down")
        assert self.names(library, f1.id) == ["a", "b", "c"]

    def test_shift_rejects_unknown_direction(self, setup):
        library, f1, f2, a, *_ = setup
        with pytest.raises(ValidationError):
            library.shift_video(a.id, "sideways")


def test_delete_video_removes_its_idle_jobs(library, jobs, repo, make_video):
    video = make_video()
    job = jobs.schedule(video.id)
    library.delete_video(video.id)
    assert repo.get_video(video.id) is None and repo.get_job(job.id) is None


def test_delete_video_refuses_while_running(library, jobs, make_video):
    video = make_video()
    jobs.schedule(video.id)
    jobs.claim_next()
    with pytest.raises(ConflictError):
        library.delete_video(video.id)
    assert library.get_video(video.id)


# -------------------------------------------------------------------- OCR

def blocks(*specs):
    return [RecognizedBlock(text, kind, ts) for text, kind, ts in specs]


def test_update_ocr_sorts_by_time_and_stamps_edit(library, make_video, clock):
    video = make_video(length=100)
    clock.advance(minutes=1)
    saved = library.update_ocr(video.id, blocks((" late ", "prose", 50), ("early", "equation", 5)))
    assert [(b.text, b.timestamp) for b in saved.ocr_blocks] == [("early", 5), ("late", 50)]
    assert saved.edited_at == clock.now


@pytest.mark.parametrize(
    "spec", [("", "prose", 1), ("  ", "prose", 1), ("x", "diagram", 1), ("x", "prose", -1), ("x", "prose", 500)]
)
def test_update_ocr_validation(library, make_video, spec):
    video = make_video(length=100)
    with pytest.raises(ValidationError):
        library.update_ocr(video.id, blocks(spec))


def test_update_ocr_can_clear_all_blocks(library, make_video):
    video = make_video(length=100)
    library.update_ocr(video.id, blocks(("x", "prose", 1)))
    assert library.update_ocr(video.id, []).ocr_blocks == ()


def test_record_processing_result(library, make_video, clock):
    video = make_video(length=100)
    clock.advance(minutes=3)
    done = library.record_processing_result(video.id, blocks(("b", "prose", 9), ("a", "prose", 1)))
    assert done.processed_at == clock.now and [b.text for b in done.ocr_blocks] == ["a", "b"]
    assert library.record_processing_result("missing", []) is None


def test_record_export_and_last_activity(library, make_video, clock):
    video = make_video()
    clock.advance(hours=2)
    exported = library.record_export(video.id)
    assert exported.exported_at == clock.now and exported.last_activity_at == clock.now
    assert video.last_activity_at == video.uploaded_at


def test_missing_video_raises_not_found(library):
    with pytest.raises(NotFoundError):
        library.get_video("missing")
    assert library.find_video("missing") is None
