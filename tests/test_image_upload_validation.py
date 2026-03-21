from ai_agent.utils import is_file_size_valid, MAX_IMAGE_SIZE_BYTES


def test_is_file_size_valid_under_limit():
    """制限内のファイルサイズに対してTrueを返すこと"""
    assert is_file_size_valid(MAX_IMAGE_SIZE_BYTES - 1) is True


def test_is_file_size_valid_at_limit():
    """制限ぴったりのファイルサイズに対してTrueを返すこと"""
    assert is_file_size_valid(MAX_IMAGE_SIZE_BYTES) is True


def test_is_file_size_valid_over_limit():
    """制限を超えるファイルサイズに対してFalseを返すこと"""
    assert is_file_size_valid(MAX_IMAGE_SIZE_BYTES + 1) is False


def test_is_file_size_valid_zero():
    """サイズ0に対してTrueを返すこと"""
    assert is_file_size_valid(0) is True

if __name__ == "__main__":
    test_is_file_size_valid_under_limit()
    test_is_file_size_valid_at_limit()
    test_is_file_size_valid_over_limit()
    test_is_file_size_valid_zero()
    print("All tests passed!")
