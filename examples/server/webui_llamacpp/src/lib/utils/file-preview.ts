<<<<<<< Updated upstream
/**
 * Formats file size in bytes to human readable format
 * @param bytes - File size in bytes
 * @returns Formatted file size string
 */
export function formatFileSize(bytes: number): string {
	if (bytes === 0) return '0 Bytes';

	const k = 1024;
	const sizes = ['Bytes', 'KB', 'MB', 'GB'];
	const i = Math.floor(Math.log(bytes) / Math.log(k));

	return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Gets a display label for a file type
 * @param fileType - The file type/mime type
 * @returns Formatted file type label
 */
export function getFileTypeLabel(fileType: string): string {
	return fileType.split('/').pop()?.toUpperCase() || 'FILE';
}

/**
 * Truncates text content for preview display
 * @param content - The text content to truncate
 * @returns Truncated content with ellipsis if needed
 */
export function getPreviewText(content: string): string {
	return content.length > 150 ? content.substring(0, 150) + '...' : content;
}
=======
/**
 * Gets a display label for a file type from various input formats
 *
 * Handles:
 * - MIME types: 'application/pdf' → 'PDF'
 * - AttachmentType values: 'PDF', 'AUDIO' → 'PDF', 'AUDIO'
 * - File names: 'document.pdf' → 'PDF'
 * - Unknown: returns 'FILE'
 *
 * @param input - MIME type, AttachmentType value, or file name
 * @returns Formatted file type label (uppercase)
 */
export function getFileTypeLabel(input: string | undefined): string {
	if (!input) return 'FILE';

	// Handle MIME types (contains '/')
	if (input.includes('/')) {
		const subtype = input.split('/').pop();
		if (subtype) {
			// Handle special cases like 'vnd.ms-excel' → 'EXCEL'
			if (subtype.includes('.')) {
				return subtype.split('.').pop()?.toUpperCase() || 'FILE';
			}
			return subtype.toUpperCase();
		}
	}

	// Handle file names (contains '.')
	if (input.includes('.')) {
		const ext = input.split('.').pop();
		if (ext) return ext.toUpperCase();
	}

	// Handle AttachmentType or other plain strings
	return input.toUpperCase();
}
>>>>>>> Stashed changes
