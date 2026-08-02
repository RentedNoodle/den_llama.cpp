<<<<<<< Updated upstream:examples/server/webui_llamacpp/src/lib/components/app/chat/ChatAttachments/ChatAttachmentThumbnailImage.svelte
<script lang="ts">
	import { RemoveButton } from '$lib/components/app';

	interface Props {
		id: string;
		name: string;
		preview: string;
		readonly?: boolean;
		onRemove?: (id: string) => void;
		onClick?: (event?: MouseEvent) => void;
		class?: string;
		// Customizable size props
		width?: string;
		height?: string;
		imageClass?: string;
	}

	let {
		id,
		name,
		preview,
		readonly = false,
		onRemove,
		onClick,
		class: className = '',
		// Default to small size for form previews
		width = 'w-auto',
		height = 'h-16',
		imageClass = ''
	}: Props = $props();
</script>

<div class="group relative overflow-hidden rounded-lg border border-border bg-muted {className}">
	{#if onClick}
		<button
			type="button"
			class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			onclick={onClick}
			aria-label="Preview {name}"
		>
			<img
				src={preview}
				alt={name}
				class="{height} {width} cursor-pointer object-cover {imageClass}"
			/>
		</button>
	{:else}
		<img
			src={preview}
			alt={name}
			class="{height} {width} cursor-pointer object-cover {imageClass}"
		/>
	{/if}

	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<RemoveButton {id} {onRemove} class="text-white" />
		</div>
	{/if}
</div>
=======
<script lang="ts">
	import { ActionIcon } from '$lib/components/app';
	import { X } from '@lucide/svelte';

	interface Props {
		class?: string;
		height?: string;
		id: string;
		imageClass?: string;
		onclick?: (event?: MouseEvent) => void;
		onRemove?: (id: string) => void;
		name: string;
		preview: string;
		readonly?: boolean;
		width?: string;
	}

	let {
		class: className = '',
		height = 'h-16',
		id,
		imageClass = '',
		onclick,
		onRemove,
		name,
		preview,
		readonly = false,
		width = 'w-auto'
	}: Props = $props();
</script>

{#snippet image()}
	<img src={preview} alt={name} class="{height} {width} cursor-pointer object-cover {imageClass}" />
{/snippet}

<div
	class="group relative overflow-hidden rounded-lg bg-muted shadow-lg dark:border dark:border-muted {className}"
>
	{#if onclick}
		<button
			aria-label="Preview {name}"
			class="block h-full w-full rounded-lg focus:ring-2 focus:ring-primary focus:ring-offset-2 focus:outline-none"
			{onclick}
			type="button"
		>
			{@render image()}
		</button>
	{:else}
		{@render image()}
	{/if}

	{#if !readonly}
		<div
			class="absolute top-1 right-1 flex items-center justify-center opacity-0 transition-opacity group-hover:opacity-100"
		>
			<ActionIcon
				class="text-white"
				icon={X}
				onclick={() => onRemove?.(id)}
				stopPropagationOnClick
				tooltip="Remove"
			/>
		</div>
	{/if}
</div>
>>>>>>> Stashed changes:examples/server/webui_llamacpp/src/lib/components/app/chat/ChatAttachments/ChatAttachmentsList/ChatAttachmentsListItem/ChatAttachmentsListItemThumbnailImage.svelte
