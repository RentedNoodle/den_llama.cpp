<<<<<<< Updated upstream
<script lang="ts">
	import { PROCESSING_INFO_TIMEOUT } from '$lib/constants/processing-info';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { slotsService } from '$lib/services/slots';
	import { isLoading, activeMessages, activeConversation } from '$lib/stores/chat.svelte';
	import { config } from '$lib/stores/settings.svelte';

	const processingState = useProcessingState();

	let isCurrentConversationLoading = $derived(isLoading());
	let processingDetails = $derived(processingState.getProcessingDetails());
	let showSlotsInfo = $derived(isCurrentConversationLoading || config().keepStatsVisible);

	// Track loading state reactively by checking if conversation ID is in loading conversations array
	$effect(() => {
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible || isCurrentConversationLoading) {
			processingState.startMonitoring();
		}

		if (!isCurrentConversationLoading && !keepStatsVisible) {
			setTimeout(() => {
				if (!config().keepStatsVisible) {
					processingState.stopMonitoring();
				}
			}, PROCESSING_INFO_TIMEOUT);
		}
	});

	// Update processing state from stored timings
	$effect(() => {
		const conversation = activeConversation();
		const messages = activeMessages() as DatabaseMessage[];
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible && conversation) {
			if (messages.length === 0) {
				slotsService.clearConversationState(conversation.id);
				return;
			}

			// Search backwards through messages to find most recent assistant message with timing data
			// Using reverse iteration for performance - avoids array copy and stops at first match
			let foundTimingData = false;

			for (let i = messages.length - 1; i >= 0; i--) {
				const message = messages[i];
				if (message.role === 'assistant' && message.timings) {
					foundTimingData = true;

					slotsService
						.updateFromTimingData(
							{
								prompt_n: message.timings.prompt_n || 0,
								predicted_n: message.timings.predicted_n || 0,
								predicted_per_second:
									message.timings.predicted_n && message.timings.predicted_ms
										? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
										: 0,
								cache_n: message.timings.cache_n || 0
							},
							conversation.id
						)
						.catch((error) => {
							console.warn('Failed to update processing state from stored timings:', error);
						});
					break;
				}
			}

			if (!foundTimingData) {
				slotsService.clearConversationState(conversation.id);
			}
		}
	});
</script>

<div class="chat-processing-info-container pointer-events-none" class:visible={showSlotsInfo}>
	<div class="chat-processing-info-content">
		{#each processingDetails as detail (detail)}
			<span class="chat-processing-info-detail pointer-events-auto">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.chat-processing-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		padding: 1.5rem 1rem;
		opacity: 0;
		transform: translateY(50%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.chat-processing-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.chat-processing-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.chat-processing-info-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		background: var(--muted);
		border-radius: 0.375rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
		white-space: nowrap;
	}

	@media (max-width: 768px) {
		.chat-processing-info-content {
			gap: 0.5rem;
		}

		.chat-processing-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
=======
<script lang="ts">
	import { untrack } from 'svelte';
	import { PROCESSING_INFO_TIMEOUT } from '$lib/constants';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { chatStore, isLoading, isChatStreaming } from '$lib/stores/chat.svelte';
	import { activeMessages, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { getProcessingInfoContext } from '$lib/contexts';
	import { page } from '$app/state';

	const processingState = useProcessingState();
	const processingInfoCtx = getProcessingInfoContext();

	let showProcessingInfo = $derived(processingInfoCtx.showProcessingInfo);

	let isCurrentConversationLoading = $derived(isLoading());
	let isStreaming = $derived(isChatStreaming());
	let processingDetails = $derived(processingState.getTechnicalDetails());

	let processingVisible = $derived(processingDetails.length > 0);

	let { onVisibilityChange }: { onVisibilityChange?: (visible: boolean) => void } = $props();

	$effect(() => {
		onVisibilityChange?.(processingVisible);
	});

	$effect(() => {
		const conversation = activeConversation();

		untrack(() => chatStore.setActiveProcessingConversation(conversation?.id ?? null));
	});

	$effect(() => {
		const keepStatsVisible = config().keepStatsVisible;
		const shouldMonitor = keepStatsVisible || isCurrentConversationLoading || isStreaming;

		if (shouldMonitor) {
			processingState.startMonitoring();
		}

		if (!isCurrentConversationLoading && !isStreaming && !keepStatsVisible) {
			const timeout = setTimeout(() => {
				if (!config().keepStatsVisible && !isChatStreaming()) {
					processingState.stopMonitoring();
				}
			}, PROCESSING_INFO_TIMEOUT);

			return () => clearTimeout(timeout);
		}
	});

	$effect(() => {
		const conversation = activeConversation();
		const messages = activeMessages() as DatabaseMessage[];
		const keepStatsVisible = config().keepStatsVisible;

		if (keepStatsVisible && conversation) {
			if (messages.length === 0) {
				untrack(() => chatStore.clearProcessingState(conversation.id));
				return;
			}

			if (!isCurrentConversationLoading && !isStreaming) {
				untrack(() => chatStore.restoreProcessingStateFromMessages(messages, conversation.id));
			}
		}
	});
</script>

<div
	class={[
		'chat-processing-info-container pointer-events-none relative',
		page.params.id && showProcessingInfo && 'visible'
	]}
>
	<div class="chat-processing-info-content absolute bottom-4 left-1/2 -translate-x-1/2">
		{#each processingDetails as detail (detail)}
			<span class="chat-processing-info-detail pointer-events-auto backdrop-blur-sm">{detail}</span>
		{/each}
	</div>
</div>

<style>
	.chat-processing-info-container {
		position: sticky;
		top: 0;
		z-index: 10;
		padding: 0 1rem 0.75rem;
		opacity: 0;
		transform: translateY(50%);
		transition:
			opacity 300ms ease-out,
			transform 300ms ease-out;
	}

	.chat-processing-info-container.visible {
		opacity: 1;
		transform: translateY(0);
	}

	.chat-processing-info-content {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 1rem;
		justify-content: center;
		max-width: 48rem;
		margin: 0 auto;
	}

	.chat-processing-info-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.75rem;
		border-radius: 0.375rem;
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Consolas, 'Liberation Mono', Menlo, monospace;
		white-space: nowrap;
	}

	@media (max-width: 768px) {
		.chat-processing-info-content {
			gap: 0.5rem;
		}

		.chat-processing-info-detail {
			font-size: 0.7rem;
			padding: 0.2rem 0.5rem;
		}
	}
</style>
>>>>>>> Stashed changes
