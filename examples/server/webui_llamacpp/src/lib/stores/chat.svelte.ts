<<<<<<< Updated upstream
import { DatabaseStore } from '$lib/stores/database';
import { chatService, slotsService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { normalizeModelName } from '$lib/utils/model-names';
import { filterByLeafNodeId, findLeafNode, findDescendantMessages } from '$lib/utils/branching';
import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { toast } from 'svelte-sonner';
import { SvelteMap } from 'svelte/reactivity';
import type { ExportedConversations } from '$lib/types/database';

/**
 * ChatStore - Central state management for chat conversations and AI interactions
 *
 * This store manages the complete chat experience including:
 * - Conversation lifecycle (create, load, delete, update)
 * - Message management with branching support for conversation trees
 * - Real-time AI response streaming with reasoning content support
 * - File attachment handling and processing
 * - Context error management and recovery
 * - Database persistence through DatabaseStore integration
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Handles low-level API communication with AI models
 *   - ChatStore orchestrates ChatService for streaming responses
 *   - ChatService provides abort capabilities and error handling
 *   - ChatStore manages the UI state while ChatService handles network layer
 *
 * - **DatabaseStore**: Provides persistent storage for conversations and messages
 *   - ChatStore uses DatabaseStore for all CRUD operations
 *   - Maintains referential integrity for conversation trees
 *   - Handles message branching and parent-child relationships
 *
 * - **SlotsService**: Monitors server resource usage during AI generation
 *   - ChatStore coordinates slots polling during streaming
 *   - Provides real-time feedback on server capacity
 *
 * **Key Features:**
 * - Reactive state management using Svelte 5 runes ($state)
 * - Conversation branching for exploring different response paths
 * - Streaming AI responses with real-time content updates
 * - File attachment support (images, PDFs, text files, audio)
 * - Partial response saving when generation is interrupted
 * - Message editing with automatic response regeneration
 */
class ChatStore {
	activeConversation = $state<DatabaseConversation | null>(null);
	activeMessages = $state<DatabaseMessage[]>([]);
	conversations = $state<DatabaseConversation[]>([]);
	currentResponse = $state('');
	errorDialogState = $state<{ type: 'timeout' | 'server'; message: string } | null>(null);
	isInitialized = $state(false);
	isLoading = $state(false);
	conversationLoadingStates = new SvelteMap<string, boolean>();
	conversationStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	titleUpdateConfirmationCallback?: (currentTitle: string, newTitle: string) => Promise<boolean>;

	constructor() {
		if (browser) {
			this.initialize();
		}
	}

	/**
	 * Initializes the chat store by loading conversations from the database
	 * Sets up the initial state and loads existing conversations
	 */
	async initialize(): Promise<void> {
		try {
			await this.loadConversations();

			this.isInitialized = true;
		} catch (error) {
			console.error('Failed to initialize chat store:', error);
		}
	}

	/**
	 * Loads all conversations from the database
	 * Refreshes the conversations list from persistent storage
	 */
	async loadConversations(): Promise<void> {
		this.conversations = await DatabaseStore.getAllConversations();
	}

	/**
	 * Creates a new conversation and navigates to it
	 * @param name - Optional name for the conversation, defaults to timestamped name
	 * @returns The ID of the created conversation
	 */
	async createConversation(name?: string): Promise<string> {
		const conversationName = name || `Chat ${new Date().toLocaleString()}`;
		const conversation = await DatabaseStore.createConversation(conversationName);

		this.conversations.unshift(conversation);

		this.activeConversation = conversation;
		this.activeMessages = [];

		slotsService.setActiveConversation(conversation.id);

		const isConvLoading = this.isConversationLoading(conversation.id);
		this.isLoading = isConvLoading;

		this.currentResponse = '';

		await goto(`#/chat/${conversation.id}`);

		return conversation.id;
	}

	/**
	 * Loads a specific conversation and its messages
	 * @param convId - The conversation ID to load
	 * @returns True if conversation was loaded successfully, false otherwise
	 */
	async loadConversation(convId: string): Promise<boolean> {
		try {
			const conversation = await DatabaseStore.getConversation(convId);

			if (!conversation) {
				return false;
			}

			this.activeConversation = conversation;

			slotsService.setActiveConversation(convId);

			const isConvLoading = this.isConversationLoading(convId);
			this.isLoading = isConvLoading;

			const streamingState = this.getConversationStreaming(convId);
			this.currentResponse = streamingState?.response || '';

			if (conversation.currNode) {
				const allMessages = await DatabaseStore.getConversationMessages(convId);
				this.activeMessages = filterByLeafNodeId(
					allMessages,
					conversation.currNode,
					false
				) as DatabaseMessage[];
			} else {
				// Load all messages for conversations without currNode (backward compatibility)
				this.activeMessages = await DatabaseStore.getConversationMessages(convId);
			}

			return true;
		} catch (error) {
			console.error('Failed to load conversation:', error);

			return false;
		}
	}

	/**
	 * Adds a new message to the active conversation
	 * @param role - The role of the message sender (user/assistant)
	 * @param content - The message content
	 * @param type - The message type, defaults to 'text'
	 * @param parent - Parent message ID, defaults to '-1' for auto-detection
	 * @param extras - Optional extra data (files, attachments, etc.)
	 * @returns The created message or null if failed
	 */
	async addMessage(
		role: ChatRole,
		content: string,
		type: ChatMessageType = 'text',
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage | null> {
		if (!this.activeConversation) {
			console.error('No active conversation when trying to add message');
			return null;
		}

		try {
			let parentId: string | null = null;

			if (parent === '-1') {
				if (this.activeMessages.length > 0) {
					parentId = this.activeMessages[this.activeMessages.length - 1].id;
				} else {
					const allMessages = await DatabaseStore.getConversationMessages(
						this.activeConversation.id
					);
					const rootMessage = allMessages.find((m) => m.parent === null && m.type === 'root');

					if (!rootMessage) {
						const rootId = await DatabaseStore.createRootMessage(this.activeConversation.id);
						parentId = rootId;
					} else {
						parentId = rootMessage.id;
					}
				}
			} else {
				parentId = parent;
			}

			const message = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					role,
					content,
					type,
					timestamp: Date.now(),
					thinking: '',
					toolCalls: '',
					children: [],
					extra: extras
				},
				parentId
			);

			this.activeMessages.push(message);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, message.id);
			this.activeConversation.currNode = message.id;

			this.updateConversationTimestamp();

			return message;
		} catch (error) {
			console.error('Failed to add message:', error);
			return null;
		}
	}

	/**
	 * Gets API options from current configuration settings
	 * Converts settings store values to API-compatible format
	 * @returns API options object for chat completion requests
	 */
	private getApiOptions(): Record<string, unknown> {
		const currentConfig = config();
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';

		const apiOptions: Record<string, unknown> = {
			stream: true,
			timings_per_token: true
		};

		if (hasValue(currentConfig.temperature)) {
			apiOptions.temperature = Number(currentConfig.temperature);
		}
		if (hasValue(currentConfig.max_tokens)) {
			apiOptions.max_tokens = Number(currentConfig.max_tokens);
		}
		if (hasValue(currentConfig.dynatemp_range)) {
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);
		}
		if (hasValue(currentConfig.dynatemp_exponent)) {
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);
		}
		if (hasValue(currentConfig.top_k)) {
			apiOptions.top_k = Number(currentConfig.top_k);
		}
		if (hasValue(currentConfig.top_p)) {
			apiOptions.top_p = Number(currentConfig.top_p);
		}
		if (hasValue(currentConfig.min_p)) {
			apiOptions.min_p = Number(currentConfig.min_p);
		}
		if (hasValue(currentConfig.xtc_probability)) {
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);
		}
		if (hasValue(currentConfig.xtc_threshold)) {
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);
		}
		if (hasValue(currentConfig.typ_p)) {
			apiOptions.typ_p = Number(currentConfig.typ_p);
		}
		if (hasValue(currentConfig.repeat_last_n)) {
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);
		}
		if (hasValue(currentConfig.repeat_penalty)) {
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);
		}
		if (hasValue(currentConfig.presence_penalty)) {
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);
		}
		if (hasValue(currentConfig.frequency_penalty)) {
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);
		}
		if (hasValue(currentConfig.dry_multiplier)) {
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);
		}
		if (hasValue(currentConfig.dry_base)) {
			apiOptions.dry_base = Number(currentConfig.dry_base);
		}
		if (hasValue(currentConfig.dry_allowed_length)) {
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);
		}
		if (hasValue(currentConfig.dry_penalty_last_n)) {
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);
		}
		if (currentConfig.samplers) {
			apiOptions.samplers = currentConfig.samplers;
		}
		if (currentConfig.custom) {
			apiOptions.custom = currentConfig.custom;
		}

		return apiOptions;
	}

	/**
	 * Helper methods for per-conversation loading state management
	 */
	private setConversationLoading(convId: string, loading: boolean): void {
		if (loading) {
			this.conversationLoadingStates.set(convId, true);
			if (this.activeConversation?.id === convId) {
				this.isLoading = true;
			}
		} else {
			this.conversationLoadingStates.delete(convId);
			if (this.activeConversation?.id === convId) {
				this.isLoading = false;
			}
		}
	}

	private isConversationLoading(convId: string): boolean {
		return this.conversationLoadingStates.get(convId) || false;
	}

	private setConversationStreaming(convId: string, response: string, messageId: string): void {
		this.conversationStreamingStates.set(convId, { response, messageId });
		if (this.activeConversation?.id === convId) {
			this.currentResponse = response;
		}
	}

	private clearConversationStreaming(convId: string): void {
		this.conversationStreamingStates.delete(convId);
		if (this.activeConversation?.id === convId) {
			this.currentResponse = '';
		}
	}

	private getConversationStreaming(
		convId: string
	): { response: string; messageId: string } | undefined {
		return this.conversationStreamingStates.get(convId);
	}

	/**
	 * Handles streaming chat completion with the AI model
	 * @param allMessages - All messages in the conversation
	 * @param assistantMessage - The assistant message to stream content into
	 * @param onComplete - Optional callback when streaming completes
	 * @param onError - Optional callback when an error occurs
	 */
	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void
	): Promise<void> {
		let streamedContent = '';
		let streamedReasoningContent = '';
		let streamedToolCallContent = '';

		let resolvedModel: string | null = null;
		let modelPersisted = false;
		const currentConfig = config();
		const preferServerPropsModel = !currentConfig.modelSelectorEnabled;
		let serverPropsRefreshed = false;
		let updateModelFromServerProps: ((persistImmediately?: boolean) => void) | null = null;

		const refreshServerPropsOnce = () => {
			if (serverPropsRefreshed) {
				return;
			}

			serverPropsRefreshed = true;

			const hasExistingProps = serverStore.serverProps !== null;

			serverStore
				.fetchServerProps({ silent: hasExistingProps })
				.then(() => {
					updateModelFromServerProps?.(true);
				})
				.catch((error) => {
					console.warn('Failed to refresh server props after streaming started:', error);
				});
		};

		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			const serverModelName = serverStore.modelName;
			const preferredModelSource = preferServerPropsModel
				? (serverModelName ?? modelName ?? null)
				: (modelName ?? serverModelName ?? null);

			if (!preferredModelSource) {
				return;
			}

			const normalizedModel = normalizeModelName(preferredModelSource);

			if (!normalizedModel || normalizedModel === resolvedModel) {
				return;
			}

			resolvedModel = normalizedModel;

			const messageIndex = this.findMessageIndex(assistantMessage.id);

			this.updateMessageAtIndex(messageIndex, { model: normalizedModel });

			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseStore.updateMessage(assistantMessage.id, { model: normalizedModel }).catch(
					(error) => {
						console.error('Failed to persist model name:', error);
						modelPersisted = false;
						resolvedModel = null;
					}
				);
			}
		};

		if (preferServerPropsModel) {
			updateModelFromServerProps = (persistImmediately = true) => {
				const currentServerModel = serverStore.modelName;

				if (!currentServerModel) {
					return;
				}

				recordModel(currentServerModel, persistImmediately);
			};

			updateModelFromServerProps(false);
		}

		slotsService.startStreaming();
		slotsService.setActiveConversation(assistantMessage.convId);

		await chatService.sendMessage(
			allMessages,
			{
				...this.getApiOptions(),

				onFirstValidChunk: () => {
					refreshServerPropsOnce();
				},
				onChunk: (chunk: string) => {
					streamedContent += chunk;
					this.setConversationStreaming(
						assistantMessage.convId,
						streamedContent,
						assistantMessage.id
					);

					const messageIndex = this.findMessageIndex(assistantMessage.id);
					this.updateMessageAtIndex(messageIndex, {
						content: streamedContent
					});
				},

				onReasoningChunk: (reasoningChunk: string) => {
					streamedReasoningContent += reasoningChunk;

					const messageIndex = this.findMessageIndex(assistantMessage.id);

					this.updateMessageAtIndex(messageIndex, { thinking: streamedReasoningContent });
				},

				onToolCallChunk: (toolCallChunk: string) => {
					const chunk = toolCallChunk.trim();

					if (!chunk) {
						return;
					}

					streamedToolCallContent = chunk;

					const messageIndex = this.findMessageIndex(assistantMessage.id);

					this.updateMessageAtIndex(messageIndex, { toolCalls: streamedToolCallContent });
				},

				onModel: (modelName: string) => {
					recordModel(modelName);
				},

				onComplete: async (
					finalContent?: string,
					reasoningContent?: string,
					timings?: ChatMessageTimings,
					toolCallContent?: string
				) => {
					slotsService.stopStreaming();

					const updateData: {
						content: string;
						thinking: string;
						toolCalls: string;
						timings?: ChatMessageTimings;
						model?: string;
					} = {
						content: finalContent || streamedContent,
						thinking: reasoningContent || streamedReasoningContent,
						toolCalls: toolCallContent || streamedToolCallContent,
						timings: timings
					};

					if (resolvedModel && !modelPersisted) {
						updateData.model = resolvedModel;
						modelPersisted = true;
					}

					await DatabaseStore.updateMessage(assistantMessage.id, updateData);

					const messageIndex = this.findMessageIndex(assistantMessage.id);

					const localUpdateData: {
						timings?: ChatMessageTimings;
						model?: string;
						toolCalls?: string;
					} = {
						timings: timings
					};

					if (updateData.model) {
						localUpdateData.model = updateData.model;
					}

					if (updateData.toolCalls !== undefined) {
						localUpdateData.toolCalls = updateData.toolCalls;
					}

					this.updateMessageAtIndex(messageIndex, localUpdateData);

					await DatabaseStore.updateCurrentNode(assistantMessage.convId, assistantMessage.id);

					if (this.activeConversation?.id === assistantMessage.convId) {
						this.activeConversation.currNode = assistantMessage.id;
						await this.refreshActiveMessages();
					}

					if (onComplete) {
						await onComplete(streamedContent);
					}

					this.setConversationLoading(assistantMessage.convId, false);
					this.clearConversationStreaming(assistantMessage.convId);
					slotsService.clearConversationState(assistantMessage.convId);
				},

				onError: (error: Error) => {
					slotsService.stopStreaming();

					if (this.isAbortError(error)) {
						this.setConversationLoading(assistantMessage.convId, false);
						this.clearConversationStreaming(assistantMessage.convId);
						slotsService.clearConversationState(assistantMessage.convId);
						return;
					}

					console.error('Streaming error:', error);
					this.setConversationLoading(assistantMessage.convId, false);
					this.clearConversationStreaming(assistantMessage.convId);
					slotsService.clearConversationState(assistantMessage.convId);

					const messageIndex = this.activeMessages.findIndex(
						(m: DatabaseMessage) => m.id === assistantMessage.id
					);

					if (messageIndex !== -1) {
						const [failedMessage] = this.activeMessages.splice(messageIndex, 1);

						if (failedMessage) {
							DatabaseStore.deleteMessage(failedMessage.id).catch((cleanupError) => {
								console.error('Failed to remove assistant message after error:', cleanupError);
							});
						}
					}

					const dialogType = error.name === 'TimeoutError' ? 'timeout' : 'server';

					this.showErrorDialog(dialogType, error.message);

					if (onError) {
						onError(error);
					}
				}
			},
			assistantMessage.convId
		);
	}

	/**
	 * Checks if an error is an abort error (user cancelled operation)
	 * @param error - The error to check
	 * @returns True if the error is an abort error
	 */
	private isAbortError(error: unknown): boolean {
		return error instanceof Error && (error.name === 'AbortError' || error instanceof DOMException);
	}

	private showErrorDialog(type: 'timeout' | 'server', message: string): void {
		this.errorDialogState = { type, message };
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	/**
	 * Finds the index of a message in the active messages array
	 * @param messageId - The message ID to find
	 * @returns The index of the message, or -1 if not found
	 */
	private findMessageIndex(messageId: string): number {
		return this.activeMessages.findIndex((m) => m.id === messageId);
	}

	/**
	 * Updates a message at a specific index with partial data
	 * @param index - The index of the message to update
	 * @param updates - Partial message data to update
	 */
	private updateMessageAtIndex(index: number, updates: Partial<DatabaseMessage>): void {
		if (index !== -1) {
			Object.assign(this.activeMessages[index], updates);
		}
	}

	/**
	 * Creates a new assistant message in the database
	 * @param parentId - Optional parent message ID, defaults to '-1'
	 * @returns The created assistant message or null if failed
	 */
	private async createAssistantMessage(parentId?: string): Promise<DatabaseMessage | null> {
		if (!this.activeConversation) return null;

		return await DatabaseStore.createMessageBranch(
			{
				convId: this.activeConversation.id,
				type: 'text',
				role: 'assistant',
				content: '',
				timestamp: Date.now(),
				thinking: '',
				toolCalls: '',
				children: [],
				model: null
			},
			parentId || null
		);
	}

	/**
	 * Updates conversation lastModified timestamp and moves it to top of list
	 * Ensures recently active conversations appear first in the sidebar
	 */
	private updateConversationTimestamp(): void {
		if (!this.activeConversation) return;

		const chatIndex = this.conversations.findIndex((c) => c.id === this.activeConversation!.id);

		if (chatIndex !== -1) {
			this.conversations[chatIndex].lastModified = Date.now();
			const updatedConv = this.conversations.splice(chatIndex, 1)[0];
			this.conversations.unshift(updatedConv);
		}
	}

	/**
	 * Sends a new message and generates AI response
	 * @param content - The message content to send
	 * @param extras - Optional extra data (files, attachments, etc.)
	 */
	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() && (!extras || extras.length === 0)) return;

		if (this.activeConversation && this.isConversationLoading(this.activeConversation.id)) {
			console.log('Cannot send message: current conversation is already processing a message');
			return;
		}

		let isNewConversation = false;

		if (!this.activeConversation) {
			await this.createConversation();
			isNewConversation = true;
		}

		if (!this.activeConversation) {
			console.error('No active conversation available for sending message');
			return;
		}

		this.errorDialogState = null;

		this.setConversationLoading(this.activeConversation.id, true);
		this.clearConversationStreaming(this.activeConversation.id);

		let userMessage: DatabaseMessage | null = null;

		try {
			userMessage = await this.addMessage('user', content, 'text', '-1', extras);

			if (!userMessage) {
				throw new Error('Failed to add user message');
			}

			if (isNewConversation && content) {
				const title = content.trim();
				await this.updateConversationName(this.activeConversation.id, title);
			}

			const assistantMessage = await this.createAssistantMessage(userMessage.id);

			if (!assistantMessage) {
				throw new Error('Failed to create assistant message');
			}

			this.activeMessages.push(assistantMessage);

			const conversationContext = this.activeMessages.slice(0, -1);

			await this.streamChatCompletion(conversationContext, assistantMessage);
		} catch (error) {
			if (this.isAbortError(error)) {
				this.setConversationLoading(this.activeConversation!.id, false);
				return;
			}

			console.error('Failed to send message:', error);
			this.setConversationLoading(this.activeConversation!.id, false);
			if (!this.errorDialogState) {
				if (error instanceof Error) {
					const dialogType = error.name === 'TimeoutError' ? 'timeout' : 'server';
					this.showErrorDialog(dialogType, error.message);
				} else {
					this.showErrorDialog('server', 'Unknown error occurred while sending message');
				}
			}
		}
	}

	/**
	 * Stops the current message generation
	 * Aborts ongoing requests and saves partial response if available
	 */
	async stopGeneration(): Promise<void> {
		if (!this.activeConversation) return;

		const convId = this.activeConversation.id;

		await this.savePartialResponseIfNeeded(convId);

		slotsService.stopStreaming();
		chatService.abort(convId);

		this.setConversationLoading(convId, false);
		this.clearConversationStreaming(convId);
		slotsService.clearConversationState(convId);
	}

	/**
	 * Gracefully stops generation and saves partial response
	 */
	async gracefulStop(): Promise<void> {
		if (!this.isLoading) return;

		slotsService.stopStreaming();
		chatService.abort();
		await this.savePartialResponseIfNeeded();

		this.conversationLoadingStates.clear();
		this.conversationStreamingStates.clear();
		this.isLoading = false;
		this.currentResponse = '';
	}

	/**
	 * Saves partial response if generation was interrupted
	 * Preserves user's partial content and timing data when generation is stopped early
	 */
	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || this.activeConversation?.id;
		if (!conversationId) return;

		const streamingState = this.conversationStreamingStates.get(conversationId);
		if (!streamingState || !streamingState.response.trim()) {
			return;
		}

		const messages =
			conversationId === this.activeConversation?.id
				? this.activeMessages
				: await DatabaseStore.getConversationMessages(conversationId);

		if (!messages.length) return;

		const lastMessage = messages[messages.length - 1];

		if (lastMessage && lastMessage.role === 'assistant') {
			try {
				const updateData: {
					content: string;
					thinking?: string;
					timings?: ChatMessageTimings;
				} = {
					content: streamingState.response
				};

				if (lastMessage.thinking?.trim()) {
					updateData.thinking = lastMessage.thinking;
				}

				const lastKnownState = await slotsService.getCurrentState();

				if (lastKnownState) {
					updateData.timings = {
						prompt_n: lastKnownState.promptTokens || 0,
						predicted_n: lastKnownState.tokensDecoded || 0,
						cache_n: lastKnownState.cacheTokens || 0,
						predicted_ms:
							lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
								? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
								: undefined
					};
				}

				await DatabaseStore.updateMessage(lastMessage.id, updateData);

				lastMessage.content = this.currentResponse;
				if (updateData.thinking !== undefined) {
					lastMessage.thinking = updateData.thinking;
				}
				if (updateData.timings) {
					lastMessage.timings = updateData.timings;
				}
			} catch (error) {
				lastMessage.content = this.currentResponse;
				console.error('Failed to save partial response:', error);
			}
		} else {
			console.error('Last message is not an assistant message');
		}
	}

	/**
	 * Updates a user message and regenerates the assistant response
	 * @param messageId - The ID of the message to update
	 * @param newContent - The new content for the message
	 */
	async updateMessage(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation) return;

		if (this.isLoading) {
			this.stopGeneration();
		}

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for update');
				return;
			}

			const messageToUpdate = this.activeMessages[messageIndex];
			const originalContent = messageToUpdate.content;

			if (messageToUpdate.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				rootMessage && messageToUpdate.parent === rootMessage.id && messageToUpdate.role === 'user';

			this.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseStore.updateMessage(messageId, { content: newContent });

			if (isFirstUserMessage && newContent.trim()) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newContent.trim(),
					this.titleUpdateConfirmationCallback
				);
			}

			const messagesToRemove = this.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) {
				await DatabaseStore.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex + 1);
			this.updateConversationTimestamp();

			this.setConversationLoading(this.activeConversation.id, true);
			this.clearConversationStreaming(this.activeConversation.id);

			try {
				const assistantMessage = await this.createAssistantMessage();
				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				this.activeMessages.push(assistantMessage);
				await DatabaseStore.updateCurrentNode(this.activeConversation.id, assistantMessage.id);
				this.activeConversation.currNode = assistantMessage.id;

				await this.streamChatCompletion(
					this.activeMessages.slice(0, -1),
					assistantMessage,
					undefined,
					() => {
						const editedMessageIndex = this.findMessageIndex(messageId);
						this.updateMessageAtIndex(editedMessageIndex, { content: originalContent });
					}
				);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.setConversationLoading(this.activeConversation!.id, false);

				const messageIndex = this.findMessageIndex(messageId);
				this.updateMessageAtIndex(messageIndex, { content: originalContent });
			}
		} catch (error) {
			if (this.isAbortError(error)) {
				return;
			}

			console.error('Failed to update message:', error);
		}
	}

	/**
	 * Regenerates an assistant message with a new response
	 * @param messageId - The ID of the assistant message to regenerate
	 */
	async regenerateMessage(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeMessages[messageIndex];
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			const messagesToRemove = this.activeMessages.slice(messageIndex);
			for (const message of messagesToRemove) {
				await DatabaseStore.deleteMessage(message.id);
			}

			this.activeMessages = this.activeMessages.slice(0, messageIndex);
			this.updateConversationTimestamp();

			this.setConversationLoading(this.activeConversation.id, true);
			this.clearConversationStreaming(this.activeConversation.id);

			try {
				const parentMessageId =
					this.activeMessages.length > 0
						? this.activeMessages[this.activeMessages.length - 1].id
						: null;

				const assistantMessage = await this.createAssistantMessage(parentMessageId);

				if (!assistantMessage) {
					throw new Error('Failed to create assistant message');
				}

				this.activeMessages.push(assistantMessage);

				const conversationContext = this.activeMessages.slice(0, -1);

				await this.streamChatCompletion(conversationContext, assistantMessage);
			} catch (regenerateError) {
				console.error('Failed to regenerate response:', regenerateError);
				this.setConversationLoading(this.activeConversation!.id, false);
			}
		} catch (error) {
			if (this.isAbortError(error)) return;
			console.error('Failed to regenerate message:', error);
		}
	}

	/**
	 * Updates the name of a conversation
	 * @param convId - The conversation ID to update
	 * @param name - The new name for the conversation
	 */
	async updateConversationName(convId: string, name: string): Promise<void> {
		try {
			await DatabaseStore.updateConversation(convId, { name });

			const convIndex = this.conversations.findIndex((c) => c.id === convId);

			if (convIndex !== -1) {
				this.conversations[convIndex].name = name;
			}

			if (this.activeConversation?.id === convId) {
				this.activeConversation.name = name;
			}
		} catch (error) {
			console.error('Failed to update conversation name:', error);
		}
	}

	/**
	 * Sets the callback function for title update confirmations
	 * @param callback - Function to call when confirmation is needed
	 */
	setTitleUpdateConfirmationCallback(
		callback: (currentTitle: string, newTitle: string) => Promise<boolean>
	): void {
		this.titleUpdateConfirmationCallback = callback;
	}

	/**
	 * Updates conversation title with optional confirmation dialog based on settings
	 * @param convId - The conversation ID to update
	 * @param newTitle - The new title content
	 * @param onConfirmationNeeded - Callback when user confirmation is needed
	 * @returns Promise<boolean> - True if title was updated, false if cancelled
	 */
	async updateConversationTitleWithConfirmation(
		convId: string,
		newTitle: string,
		onConfirmationNeeded?: (currentTitle: string, newTitle: string) => Promise<boolean>
	): Promise<boolean> {
		try {
			const currentConfig = config();

			if (currentConfig.askForTitleConfirmation && onConfirmationNeeded) {
				const conversation = await DatabaseStore.getConversation(convId);
				if (!conversation) return false;

				const shouldUpdate = await onConfirmationNeeded(conversation.name, newTitle);
				if (!shouldUpdate) return false;
			}

			await this.updateConversationName(convId, newTitle);
			return true;
		} catch (error) {
			console.error('Failed to update conversation title with confirmation:', error);
			return false;
		}
	}

	/**
	 * Downloads a conversation as JSON file
	 * @param convId - The conversation ID to download
	 */
	async downloadConversation(convId: string): Promise<void> {
		if (!this.activeConversation || this.activeConversation.id !== convId) {
			// Load the conversation if not currently active
			const conversation = await DatabaseStore.getConversation(convId);
			if (!conversation) return;

			const messages = await DatabaseStore.getConversationMessages(convId);
			const conversationData = {
				conv: conversation,
				messages
			};

			this.triggerDownload(conversationData);
		} else {
			// Use current active conversation data
			const conversationData: ExportedConversations = {
				conv: this.activeConversation!,
				messages: this.activeMessages
			};

			this.triggerDownload(conversationData);
		}
	}

	/**
	 * Triggers file download in browser
	 * @param data - Data to download (expected: { conv: DatabaseConversation, messages: DatabaseMessage[] })
	 * @param filename - Optional filename
	 */
	private triggerDownload(data: ExportedConversations, filename?: string): void {
		const conversation =
			'conv' in data ? data.conv : Array.isArray(data) ? data[0]?.conv : undefined;
		if (!conversation) {
			console.error('Invalid data: missing conversation');
			return;
		}
		const conversationName = conversation.name ? conversation.name.trim() : '';
		const convId = conversation.id || 'unknown';
		const truncatedSuffix = conversationName
			.toLowerCase()
			.replace(/[^a-z0-9]/gi, '_')
			.replace(/_+/g, '_')
			.substring(0, 20);
		const downloadFilename = filename || `conversation_${convId}_${truncatedSuffix}.json`;

		const conversationJson = JSON.stringify(data, null, 2);
		const blob = new Blob([conversationJson], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = downloadFilename;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	}

	/**
	 * Exports all conversations with their messages as a JSON file
	 * Returns the list of exported conversations
	 */
	async exportAllConversations(): Promise<DatabaseConversation[]> {
		try {
			const allConversations = await DatabaseStore.getAllConversations();
			if (allConversations.length === 0) {
				throw new Error('No conversations to export');
			}

			const allData: ExportedConversations = await Promise.all(
				allConversations.map(async (conv) => {
					const messages = await DatabaseStore.getConversationMessages(conv.id);
					return { conv, messages };
				})
			);

			const blob = new Blob([JSON.stringify(allData, null, 2)], {
				type: 'application/json'
			});
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = `all_conversations_${new Date().toISOString().split('T')[0]}.json`;
			document.body.appendChild(a);
			a.click();
			document.body.removeChild(a);
			URL.revokeObjectURL(url);

			toast.success(`All conversations (${allConversations.length}) prepared for download`);
			return allConversations;
		} catch (err) {
			console.error('Failed to export conversations:', err);
			throw err;
		}
	}

	/**
	 * Imports conversations from a JSON file.
	 * Supports both single conversation (object) and multiple conversations (array).
	 * Uses DatabaseStore for safe, encapsulated data access
	 * Returns the list of imported conversations
	 */
	async importConversations(): Promise<DatabaseConversation[]> {
		return new Promise((resolve, reject) => {
			const input = document.createElement('input');
			input.type = 'file';
			input.accept = '.json';

			input.onchange = async (e) => {
				const file = (e.target as HTMLInputElement)?.files?.[0];
				if (!file) {
					reject(new Error('No file selected'));
					return;
				}

				try {
					const text = await file.text();
					const parsedData = JSON.parse(text);
					let importedData: ExportedConversations;

					if (Array.isArray(parsedData)) {
						importedData = parsedData;
					} else if (
						parsedData &&
						typeof parsedData === 'object' &&
						'conv' in parsedData &&
						'messages' in parsedData
					) {
						// Single conversation object
						importedData = [parsedData];
					} else {
						throw new Error(
							'Invalid file format: expected array of conversations or single conversation object'
						);
					}

					const result = await DatabaseStore.importConversations(importedData);

					// Refresh UI
					await this.loadConversations();

					toast.success(`Imported ${result.imported} conversation(s), skipped ${result.skipped}`);

					// Extract the conversation objects from imported data
					const importedConversations = importedData.map((item) => item.conv);
					resolve(importedConversations);
				} catch (err: unknown) {
					const message = err instanceof Error ? err.message : 'Unknown error';
					console.error('Failed to import conversations:', err);
					toast.error('Import failed', {
						description: message
					});
					reject(new Error(`Import failed: ${message}`));
				}
			};

			input.click();
		});
	}

	/**
	 * Deletes a conversation and all its messages
	 * @param convId - The conversation ID to delete
	 */
	async deleteConversation(convId: string): Promise<void> {
		try {
			await DatabaseStore.deleteConversation(convId);

			this.conversations = this.conversations.filter((c) => c.id !== convId);

			if (this.activeConversation?.id === convId) {
				this.activeConversation = null;
				this.activeMessages = [];
				await goto(`?new_chat=true#/`);
			}
		} catch (error) {
			console.error('Failed to delete conversation:', error);
		}
	}

	/**
	 * Gets information about what messages will be deleted when deleting a specific message
	 * @param messageId - The ID of the message to be deleted
	 * @returns Object with deletion info including count and types of messages
	 */
	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		if (!this.activeConversation) {
			return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
		}

		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];

		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));

		let userMessages = 0;
		let assistantMessages = 0;
		const messageTypes: string[] = [];

		for (const msg of messagesToDelete) {
			if (msg.role === 'user') {
				userMessages++;
				if (!messageTypes.includes('user message')) messageTypes.push('user message');
			} else if (msg.role === 'assistant') {
				assistantMessages++;
				if (!messageTypes.includes('assistant response')) messageTypes.push('assistant response');
			}
		}

		return {
			totalCount: allToDelete.length,
			userMessages,
			assistantMessages,
			messageTypes
		};
	}

	/**
	 * Deletes a message and all its descendants, updating conversation path if needed
	 * @param messageId - The ID of the message to delete
	 */
	async deleteMessage(messageId: string): Promise<void> {
		try {
			if (!this.activeConversation) return;

			// Get all messages to find siblings before deletion
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const messageToDelete = allMessages.find((m) => m.id === messageId);

			if (!messageToDelete) {
				console.error('Message to delete not found');
				return;
			}

			// Check if the deleted message is in the current conversation path
			const currentPath = filterByLeafNodeId(
				allMessages,
				this.activeConversation.currNode || '',
				false
			);
			const isInCurrentPath = currentPath.some((m) => m.id === messageId);

			// If the deleted message is in the current path, we need to update currNode
			if (isInCurrentPath && messageToDelete.parent) {
				// Find all siblings (messages with same parent)
				const siblings = allMessages.filter(
					(m) => m.parent === messageToDelete.parent && m.id !== messageId
				);

				if (siblings.length > 0) {
					// Find the latest sibling (highest timestamp)
					const latestSibling = siblings.reduce((latest, sibling) =>
						sibling.timestamp > latest.timestamp ? sibling : latest
					);

					// Find the leaf node for this sibling branch to get the complete conversation path
					const leafNodeId = findLeafNode(allMessages, latestSibling.id);

					// Update conversation to use the leaf node of the latest remaining sibling
					await DatabaseStore.updateCurrentNode(this.activeConversation.id, leafNodeId);
					this.activeConversation.currNode = leafNodeId;
				} else {
					// No siblings left, navigate to parent if it exists
					if (messageToDelete.parent) {
						const parentLeafId = findLeafNode(allMessages, messageToDelete.parent);
						await DatabaseStore.updateCurrentNode(this.activeConversation.id, parentLeafId);
						this.activeConversation.currNode = parentLeafId;
					}
				}
			}

			// Use cascading deletion to remove the message and all its descendants
			await DatabaseStore.deleteMessageCascading(this.activeConversation.id, messageId);

			// Refresh active messages to show the updated branch
			await this.refreshActiveMessages();

			// Update conversation timestamp
			this.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to delete message:', error);
		}
	}

	/**
	 * Clears the active conversation and messages
	 * Used when navigating away from chat or starting fresh
	 * Note: Does not stop ongoing streaming to allow background completion
	 */
	clearActiveConversation(): void {
		this.activeConversation = null;
		this.activeMessages = [];
		this.isLoading = false;
		this.currentResponse = '';
		slotsService.setActiveConversation(null);
	}

	/** Refreshes active messages based on currNode after branch navigation */
	async refreshActiveMessages(): Promise<void> {
		if (!this.activeConversation) return;

		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		if (allMessages.length === 0) {
			this.activeMessages = [];
			return;
		}

		const leafNodeId =
			this.activeConversation.currNode ||
			allMessages.reduce((latest, msg) => (msg.timestamp > latest.timestamp ? msg : latest)).id;

		const currentPath = filterByLeafNodeId(allMessages, leafNodeId, false) as DatabaseMessage[];

		this.activeMessages.length = 0;
		this.activeMessages.push(...currentPath);
	}

	/**
	 * Navigates to a specific sibling branch by updating currNode and refreshing messages
	 * @param siblingId - The sibling message ID to navigate to
	 */
	async navigateToSibling(siblingId: string): Promise<void> {
		if (!this.activeConversation) return;

		// Get the current first user message before navigation
		const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
		const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
		const currentFirstUserMessage = this.activeMessages.find(
			(m) => m.role === 'user' && m.parent === rootMessage?.id
		);

		const currentLeafNodeId = findLeafNode(allMessages, siblingId);

		await DatabaseStore.updateCurrentNode(this.activeConversation.id, currentLeafNodeId);
		this.activeConversation.currNode = currentLeafNodeId;
		await this.refreshActiveMessages();

		// Only show title dialog if we're navigating between different first user message siblings
		if (rootMessage && this.activeMessages.length > 0) {
			// Find the first user message in the new active path
			const newFirstUserMessage = this.activeMessages.find(
				(m) => m.role === 'user' && m.parent === rootMessage.id
			);

			// Only show dialog if:
			// 1. We have a new first user message
			// 2. It's different from the previous one (different ID or content)
			// 3. The new message has content
			if (
				newFirstUserMessage &&
				newFirstUserMessage.content.trim() &&
				(!currentFirstUserMessage ||
					newFirstUserMessage.id !== currentFirstUserMessage.id ||
					newFirstUserMessage.content.trim() !== currentFirstUserMessage.content.trim())
			) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newFirstUserMessage.content.trim(),
					this.titleUpdateConfirmationCallback
				);
			}
		}
	}

	/**
	 * Edits an assistant message with optional branching
	 * @param messageId - The ID of the assistant message to edit
	 * @param newContent - The new content for the message
	 * @param shouldBranch - Whether to create a branch or replace in-place
	 */
	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);

			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];

			if (messageToEdit.role !== 'assistant') {
				console.error('Only assistant messages can be edited with this method');
				return;
			}

			if (shouldBranch) {
				const newMessage = await DatabaseStore.createMessageBranch(
					{
						convId: messageToEdit.convId,
						type: messageToEdit.type,
						timestamp: Date.now(),
						role: messageToEdit.role,
						content: newContent,
						thinking: messageToEdit.thinking || '',
						toolCalls: messageToEdit.toolCalls || '',
						children: [],
						model: messageToEdit.model // Preserve original model info when branching
					},
					messageToEdit.parent!
				);

				await DatabaseStore.updateCurrentNode(this.activeConversation.id, newMessage.id);
				this.activeConversation.currNode = newMessage.id;
			} else {
				await DatabaseStore.updateMessage(messageToEdit.id, {
					content: newContent,
					timestamp: Date.now()
				});

				// Ensure currNode points to the edited message to maintain correct path
				await DatabaseStore.updateCurrentNode(this.activeConversation.id, messageToEdit.id);
				this.activeConversation.currNode = messageToEdit.id;

				this.updateMessageAtIndex(messageIndex, {
					content: newContent,
					timestamp: Date.now()
				});
			}

			this.updateConversationTimestamp();
			await this.refreshActiveMessages();
		} catch (error) {
			console.error('Failed to edit assistant message:', error);
		}
	}

	/**
	 * Edits a user message and preserves all responses below
	 * Updates the message content in-place without deleting or regenerating responses
	 *
	 * **Use Case**: When you want to fix a typo or rephrase a question without losing the assistant's response
	 *
	 * **Important Behavior:**
	 * - Does NOT create a branch (unlike editMessageWithBranching)
	 * - Does NOT regenerate assistant responses
	 * - Only updates the user message content in the database
	 * - Preserves the entire conversation tree below the edited message
	 * - Updates conversation title if this is the first user message
	 *
	 * @param messageId - The ID of the user message to edit
	 * @param newContent - The new content for the message
	 */
	async editUserMessagePreserveResponses(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];
			if (messageToEdit.role !== 'user') {
				console.error('Only user messages can be edited with this method');
				return;
			}

			// Simply update the message content in-place
			await DatabaseStore.updateMessage(messageId, {
				content: newContent,
				timestamp: Date.now()
			});

			this.updateMessageAtIndex(messageIndex, {
				content: newContent,
				timestamp: Date.now()
			});

			// Check if first user message for title update
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				rootMessage && messageToEdit.parent === rootMessage.id && messageToEdit.role === 'user';

			if (isFirstUserMessage && newContent.trim()) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newContent.trim(),
					this.titleUpdateConfirmationCallback
				);
			}

			this.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to edit user message:', error);
		}
	}

	/**
	 * Edits a message by creating a new branch with the edited content
	 * @param messageId - The ID of the message to edit
	 * @param newContent - The new content for the message
	 */
	async editMessageWithBranching(messageId: string, newContent: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for editing');
				return;
			}

			const messageToEdit = this.activeMessages[messageIndex];
			if (messageToEdit.role !== 'user') {
				console.error('Only user messages can be edited');
				return;
			}

			// Check if this is the first user message in the conversation
			// First user message is one that has the root message as its parent
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				rootMessage && messageToEdit.parent === rootMessage.id && messageToEdit.role === 'user';

			let parentId = messageToEdit.parent;

			if (parentId === undefined || parentId === null) {
				const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
				if (rootMessage) {
					parentId = rootMessage.id;
				} else {
					console.error('No root message found for editing');
					return;
				}
			}

			const newMessage = await DatabaseStore.createMessageBranch(
				{
					convId: messageToEdit.convId,
					type: messageToEdit.type,
					timestamp: Date.now(),
					role: messageToEdit.role,
					content: newContent,
					thinking: messageToEdit.thinking || '',
					toolCalls: messageToEdit.toolCalls || '',
					children: [],
					extra: messageToEdit.extra ? JSON.parse(JSON.stringify(messageToEdit.extra)) : undefined,
					model: messageToEdit.model // Preserve original model info when branching
				},
				parentId
			);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, newMessage.id);
			this.activeConversation.currNode = newMessage.id;
			this.updateConversationTimestamp();

			// If this is the first user message, update the conversation title with confirmation if needed
			if (isFirstUserMessage && newContent.trim()) {
				await this.updateConversationTitleWithConfirmation(
					this.activeConversation.id,
					newContent.trim(),
					this.titleUpdateConfirmationCallback
				);
			}

			await this.refreshActiveMessages();

			if (messageToEdit.role === 'user') {
				await this.generateResponseForMessage(newMessage.id);
			}
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	/**
	 * Regenerates an assistant message by creating a new branch with a new response
	 * @param messageId - The ID of the assistant message to regenerate
	 */
	async regenerateMessageWithBranching(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for regeneration');
				return;
			}

			const messageToRegenerate = this.activeMessages[messageIndex];
			if (messageToRegenerate.role !== 'assistant') {
				console.error('Only assistant messages can be regenerated');
				return;
			}

			// Find parent message in all conversation messages, not just active path
			const conversationMessages = await DatabaseStore.getConversationMessages(
				this.activeConversation.id
			);
			const parentMessage = conversationMessages.find((m) => m.id === messageToRegenerate.parent);
			if (!parentMessage) {
				console.error('Parent message not found for regeneration');
				return;
			}

			this.setConversationLoading(this.activeConversation.id, true);
			this.clearConversationStreaming(this.activeConversation.id);

			const newAssistantMessage = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					type: 'text',
					timestamp: Date.now(),
					role: 'assistant',
					content: '',
					thinking: '',
					toolCalls: '',
					children: [],
					model: null
				},
				parentMessage.id
			);

			await DatabaseStore.updateCurrentNode(this.activeConversation.id, newAssistantMessage.id);
			this.activeConversation.currNode = newAssistantMessage.id;
			this.updateConversationTimestamp();
			await this.refreshActiveMessages();

			const allConversationMessages = await DatabaseStore.getConversationMessages(
				this.activeConversation.id
			);
			const conversationPath = filterByLeafNodeId(
				allConversationMessages,
				parentMessage.id,
				false
			) as DatabaseMessage[];

			await this.streamChatCompletion(conversationPath, newAssistantMessage);
		} catch (error) {
			if (this.isAbortError(error)) return;

			console.error('Failed to regenerate message with branching:', error);
			this.setConversationLoading(this.activeConversation!.id, false);
		}
	}

	/**
	 * Generates a new assistant response for a given user message
	 * @param userMessageId - ID of user message to respond to
	 */
	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		if (!this.activeConversation) return;

		this.errorDialogState = null;
		this.setConversationLoading(this.activeConversation.id, true);
		this.clearConversationStreaming(this.activeConversation.id);

		try {
			// Get conversation path up to the user message
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const conversationPath = filterByLeafNodeId(
				allMessages,
				userMessageId,
				false
			) as DatabaseMessage[];

			// Create new assistant message branch
			const assistantMessage = await DatabaseStore.createMessageBranch(
				{
					convId: this.activeConversation.id,
					type: 'text',
					timestamp: Date.now(),
					role: 'assistant',
					content: '',
					thinking: '',
					toolCalls: '',
					children: [],
					model: null
				},
				userMessageId
			);

			// Add assistant message to active messages immediately for UI reactivity
			this.activeMessages.push(assistantMessage);

			// Stream response to new assistant message
			await this.streamChatCompletion(conversationPath, assistantMessage);
		} catch (error) {
			console.error('Failed to generate response:', error);
			this.setConversationLoading(this.activeConversation!.id, false);
		}
	}

	/**
	 * Continues generation for an existing assistant message
	 * @param messageId - The ID of the assistant message to continue
	 */
	async continueAssistantMessage(messageId: string): Promise<void> {
		if (!this.activeConversation || this.isLoading) return;

		try {
			const messageIndex = this.findMessageIndex(messageId);
			if (messageIndex === -1) {
				console.error('Message not found for continuation');
				return;
			}

			const messageToContinue = this.activeMessages[messageIndex];
			if (messageToContinue.role !== 'assistant') {
				console.error('Only assistant messages can be continued');
				return;
			}

			// Race condition protection: Check if this specific conversation is already loading
			// This prevents multiple rapid clicks on "Continue" from creating concurrent operations
			if (this.isConversationLoading(this.activeConversation.id)) {
				console.warn('Continuation already in progress for this conversation');
				return;
			}

			this.errorDialogState = null;
			this.setConversationLoading(this.activeConversation.id, true);
			this.clearConversationStreaming(this.activeConversation.id);

			// IMPORTANT: Fetch the latest content from the database to ensure we have
			// the most up-to-date content, especially after a stopped generation
			// This prevents issues where the in-memory state might be stale
			const allMessages = await DatabaseStore.getConversationMessages(this.activeConversation.id);
			const dbMessage = allMessages.find((m) => m.id === messageId);

			if (!dbMessage) {
				console.error('Message not found in database for continuation');
				this.setConversationLoading(this.activeConversation.id, false);

				return;
			}

			// Use content from database as the source of truth
			const originalContent = dbMessage.content;
			const originalThinking = dbMessage.thinking || '';

			// Get conversation context up to (but not including) the message to continue
			const conversationContext = this.activeMessages.slice(0, messageIndex);

			const contextWithContinue = [
				...conversationContext.map((msg) => {
					if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
						return msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };
					}
					return msg as ApiChatMessageData;
				}),
				{
					role: 'assistant' as const,
					content: originalContent
				}
			];

			let appendedContent = '';
			let appendedThinking = '';
			let hasReceivedContent = false;

			await chatService.sendMessage(
				contextWithContinue,
				{
					...this.getApiOptions(),

					onChunk: (chunk: string) => {
						hasReceivedContent = true;
						appendedContent += chunk;
						// Preserve originalContent exactly as-is, including any trailing whitespace
						// The concatenation naturally preserves any whitespace at the end of originalContent
						const fullContent = originalContent + appendedContent;

						this.setConversationStreaming(
							messageToContinue.convId,
							fullContent,
							messageToContinue.id
						);

						this.updateMessageAtIndex(messageIndex, {
							content: fullContent
						});
					},

					onReasoningChunk: (reasoningChunk: string) => {
						hasReceivedContent = true;
						appendedThinking += reasoningChunk;

						const fullThinking = originalThinking + appendedThinking;

						this.updateMessageAtIndex(messageIndex, {
							thinking: fullThinking
						});
					},

					onComplete: async (
						finalContent?: string,
						reasoningContent?: string,
						timings?: ChatMessageTimings
					) => {
						const fullContent = originalContent + (finalContent || appendedContent);
						const fullThinking = originalThinking + (reasoningContent || appendedThinking);

						const updateData: {
							content: string;
							thinking: string;
							timestamp: number;
							timings?: ChatMessageTimings;
						} = {
							content: fullContent,
							thinking: fullThinking,
							timestamp: Date.now(),
							timings: timings
						};

						await DatabaseStore.updateMessage(messageToContinue.id, updateData);

						this.updateMessageAtIndex(messageIndex, updateData);

						this.updateConversationTimestamp();

						this.setConversationLoading(messageToContinue.convId, false);
						this.clearConversationStreaming(messageToContinue.convId);
						slotsService.clearConversationState(messageToContinue.convId);
					},

					onError: async (error: Error) => {
						if (this.isAbortError(error)) {
							// User cancelled - save partial continuation if any content was received
							if (hasReceivedContent && appendedContent) {
								const partialContent = originalContent + appendedContent;
								const partialThinking = originalThinking + appendedThinking;

								await DatabaseStore.updateMessage(messageToContinue.id, {
									content: partialContent,
									thinking: partialThinking,
									timestamp: Date.now()
								});

								this.updateMessageAtIndex(messageIndex, {
									content: partialContent,
									thinking: partialThinking,
									timestamp: Date.now()
								});
							}

							this.setConversationLoading(messageToContinue.convId, false);
							this.clearConversationStreaming(messageToContinue.convId);
							slotsService.clearConversationState(messageToContinue.convId);

							return;
						}

						// Non-abort error - rollback to original content
						console.error('Continue generation error:', error);

						// Rollback: Restore original content in UI
						this.updateMessageAtIndex(messageIndex, {
							content: originalContent,
							thinking: originalThinking
						});

						// Ensure database has original content (in case of partial writes)
						await DatabaseStore.updateMessage(messageToContinue.id, {
							content: originalContent,
							thinking: originalThinking
						});

						this.setConversationLoading(messageToContinue.convId, false);
						this.clearConversationStreaming(messageToContinue.convId);
						slotsService.clearConversationState(messageToContinue.convId);

						const dialogType = error.name === 'TimeoutError' ? 'timeout' : 'server';
						this.showErrorDialog(dialogType, error.message);
					}
				},
				messageToContinue.convId
			);
		} catch (error) {
			if (this.isAbortError(error)) return;
			console.error('Failed to continue message:', error);
			if (this.activeConversation) {
				this.setConversationLoading(this.activeConversation.id, false);
			}
		}
	}

	/**
	 * Public methods for accessing per-conversation states
	 */
	public isConversationLoadingPublic(convId: string): boolean {
		return this.isConversationLoading(convId);
	}

	public getConversationStreamingPublic(
		convId: string
	): { response: string; messageId: string } | undefined {
		return this.getConversationStreaming(convId);
	}

	public getAllLoadingConversations(): string[] {
		return Array.from(this.conversationLoadingStates.keys());
	}

	public getAllStreamingConversations(): string[] {
		return Array.from(this.conversationStreamingStates.keys());
	}
}

export const chatStore = new ChatStore();

export const conversations = () => chatStore.conversations;
export const activeConversation = () => chatStore.activeConversation;
export const activeMessages = () => chatStore.activeMessages;
export const isLoading = () => chatStore.isLoading;
export const currentResponse = () => chatStore.currentResponse;
export const isInitialized = () => chatStore.isInitialized;
export const errorDialog = () => chatStore.errorDialogState;

export const createConversation = chatStore.createConversation.bind(chatStore);
export const downloadConversation = chatStore.downloadConversation.bind(chatStore);
export const exportAllConversations = chatStore.exportAllConversations.bind(chatStore);
export const importConversations = chatStore.importConversations.bind(chatStore);
export const deleteConversation = chatStore.deleteConversation.bind(chatStore);
export const sendMessage = chatStore.sendMessage.bind(chatStore);
export const dismissErrorDialog = chatStore.dismissErrorDialog.bind(chatStore);

export const gracefulStop = chatStore.gracefulStop.bind(chatStore);

// Branching operations
export const refreshActiveMessages = chatStore.refreshActiveMessages.bind(chatStore);
export const navigateToSibling = chatStore.navigateToSibling.bind(chatStore);
export const editAssistantMessage = chatStore.editAssistantMessage.bind(chatStore);
export const editMessageWithBranching = chatStore.editMessageWithBranching.bind(chatStore);
export const editUserMessagePreserveResponses =
	chatStore.editUserMessagePreserveResponses.bind(chatStore);
export const regenerateMessageWithBranching =
	chatStore.regenerateMessageWithBranching.bind(chatStore);
export const continueAssistantMessage = chatStore.continueAssistantMessage.bind(chatStore);
export const deleteMessage = chatStore.deleteMessage.bind(chatStore);
export const getDeletionInfo = chatStore.getDeletionInfo.bind(chatStore);
export const updateConversationName = chatStore.updateConversationName.bind(chatStore);
export const setTitleUpdateConfirmationCallback =
	chatStore.setTitleUpdateConfirmationCallback.bind(chatStore);

export function stopGeneration() {
	chatStore.stopGeneration();
}
export const messages = () => chatStore.activeMessages;

// Per-conversation state access
export const isConversationLoading = (convId: string) =>
	chatStore.isConversationLoadingPublic(convId);
export const getConversationStreaming = (convId: string) =>
	chatStore.getConversationStreamingPublic(convId);
export const getAllLoadingConversations = () => chatStore.getAllLoadingConversations();
export const getAllStreamingConversations = () => chatStore.getAllStreamingConversations();
=======
/**
 * chatStore - Reactive State Store for Chat Operations
 *
 * Manages chat lifecycle, streaming, message operations, and processing state.
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **chatStore** (this): Reactive state + business logic
 * - **conversationsStore**: Conversation persistence and navigation
 *
 * @see ChatService in services/chat.service.ts for API operations
 */

import { SvelteMap } from 'svelte/reactivity';
import { DatabaseService } from '$lib/services/database.service';
import { ChatService } from '$lib/services/chat.service';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import { config } from '$lib/stores/settings.svelte';
import { agenticStore } from '$lib/stores/agentic.svelte';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { contextSize, isRouterMode } from '$lib/stores/server.svelte';
import {
	selectedModelName,
	modelsStore,
	selectedModelContextSize
} from '$lib/stores/models.svelte';
import {
	normalizeModelName,
	filterByLeafNodeId,
	findDescendantMessages,
	findLeafNode,
	findMessageById,
	isAbortError,
	generateConversationTitle
} from '$lib/utils';
import { classifyContinueIntent } from '$lib/utils/agentic';
import {
	MAX_INACTIVE_CONVERSATION_STATES,
	INACTIVE_CONVERSATION_STATE_MAX_AGE_MS,
	SYSTEM_MESSAGE_PLACEHOLDER,
	TITLE_GENERATION
} from '$lib/constants';
import type {
	ChatMessageTimings,
	ChatMessagePromptProgress,
	ChatStreamCallbacks,
	ErrorDialogState
} from '$lib/types/chat';
import type {
	ApiChatMessageData,
	ApiProcessingState,
	DatabaseMessage,
	DatabaseMessageExtra
} from '$lib/types';
import { ContinueIntentKind, ErrorDialogType, MessageRole, MessageType } from '$lib/enums';

interface ConversationStateEntry {
	lastAccessed: number;
}

class ChatStore {
	activeProcessingState = $state<ApiProcessingState | null>(null);
	currentResponse = $state('');
	errorDialogState = $state<ErrorDialogState | null>(null);
	isLoading = $state(false);
	// true while the active conversation streams reasoning content but no visible content yet
	isReasoning = $state(false);
	chatLoadingStates = new SvelteMap<string, boolean>();
	chatReasoningStates = new SvelteMap<string, boolean>();
	chatStreamingStates = new SvelteMap<string, { response: string; messageId: string }>();
	private abortControllers = new SvelteMap<string, AbortController>();
	private preEncodeAbortController: AbortController | null = null;
	private processingStates = new SvelteMap<string, ApiProcessingState | null>();
	private conversationStateTimestamps = new SvelteMap<string, ConversationStateEntry>();
	private activeConversationId = $state<string | null>(null);
	private isStreamingActive = $state(false);
	private isEditModeActive = $state(false);
	private addFilesHandler: ((files: File[]) => void) | null = $state(null);
	pendingEditMessageId = $state<string | null>(null);
	private messageUpdateCallback:
		| ((messageId: string, updates: Partial<DatabaseMessage>) => void)
		| null = null;
	private _pendingDraftMessage = $state<string>('');
	private _pendingDraftFiles = $state<ChatUploadedFile[]>([]);

	/** Reactive: queued pending messages for non-agentic streaming */
	private _pendingMessages = new SvelteMap<
		string,
		{ content: string; extras?: DatabaseMessageExtra[] }
	>();

	private setChatLoading(convId: string, loading: boolean): void {
		this.touchConversationState(convId);
		if (loading) {
			this.chatLoadingStates.set(convId, true);
			if (convId === conversationsStore.activeConversation?.id) this.isLoading = true;
		} else {
			this.chatLoadingStates.delete(convId);
			if (convId === conversationsStore.activeConversation?.id) this.isLoading = false;
			this.setChatReasoning(convId, false);
		}
	}

	private setChatReasoning(convId: string, reasoning: boolean): void {
		if (reasoning) {
			this.chatReasoningStates.set(convId, true);
			if (convId === conversationsStore.activeConversation?.id) this.isReasoning = true;
		} else {
			this.chatReasoningStates.delete(convId);
			if (convId === conversationsStore.activeConversation?.id) this.isReasoning = false;
		}
	}
	private setChatStreaming(convId: string, response: string, messageId: string): void {
		this.touchConversationState(convId);
		this.chatStreamingStates.set(convId, { response, messageId });
		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = response;
	}
	private clearChatStreaming(convId: string): void {
		this.chatStreamingStates.delete(convId);
		if (convId === conversationsStore.activeConversation?.id) this.currentResponse = '';
	}
	private getChatStreaming(convId: string): { response: string; messageId: string } | undefined {
		return this.chatStreamingStates.get(convId);
	}
	syncLoadingStateForChat(convId: string): void {
		this.isLoading = this.chatLoadingStates.get(convId) || false;
		this.isReasoning = this.chatReasoningStates.get(convId) || false;
		const s = this.chatStreamingStates.get(convId);
		this.currentResponse = s?.response || '';
		this.isStreamingActive = s !== undefined;
		this.setActiveProcessingConversation(convId);
		// Sync streaming content to activeMessages so UI displays current content
		if (s?.response && s?.messageId) {
			const idx = conversationsStore.findMessageIndex(s.messageId);
			if (idx !== -1) {
				conversationsStore.updateMessageAtIndex(idx, { content: s.response });
			}
		}
	}

	clearUIState(): void {
		this.isLoading = false;
		this.currentResponse = '';
		this.isStreamingActive = false;
	}

	setActiveProcessingConversation(conversationId: string | null): void {
		this.activeConversationId = conversationId;
		this.activeProcessingState = conversationId
			? this.processingStates.get(conversationId) || null
			: null;
	}

	getProcessingState(conversationId: string): ApiProcessingState | null {
		return this.processingStates.get(conversationId) || null;
	}

	private setProcessingState(conversationId: string, state: ApiProcessingState | null): void {
		if (state === null) this.processingStates.delete(conversationId);
		else this.processingStates.set(conversationId, state);
		if (conversationId === this.activeConversationId) this.activeProcessingState = state;
	}

	clearProcessingState(conversationId: string): void {
		this.processingStates.delete(conversationId);
		if (conversationId === this.activeConversationId) this.activeProcessingState = null;
	}

	getActiveProcessingState(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	getCurrentProcessingStateSync(): ApiProcessingState | null {
		return this.activeProcessingState;
	}

	private setStreamingActive(active: boolean): void {
		this.isStreamingActive = active;
	}

	isStreaming(): boolean {
		return this.isStreamingActive;
	}

	private getOrCreateAbortController(convId: string): AbortController {
		let c = this.abortControllers.get(convId);
		if (!c || c.signal.aborted) {
			c = new AbortController();
			this.abortControllers.set(convId, c);
		}
		return c;
	}

	private abortRequest(convId?: string): void {
		if (convId) {
			const c = this.abortControllers.get(convId);
			if (c) {
				c.abort();
				this.abortControllers.delete(convId);
			}
		} else {
			for (const c of this.abortControllers.values()) c.abort();
			this.abortControllers.clear();
		}
	}

	/**
	 * Abort the current agentic flow signal without clearing loading state.
	 * Used by "Send immediately" to force the agentic loop to exit so that
	 * the pending steering message can be re-sent.
	 */
	abortCurrentFlow(convId: string): void {
		const c = this.abortControllers.get(convId);
		if (c) {
			c.abort();
			this.abortControllers.delete(convId);
		}
	}

	private showErrorDialog(state: ErrorDialogState | null): void {
		this.errorDialogState = state;
	}

	dismissErrorDialog(): void {
		this.errorDialogState = null;
	}

	clearEditMode(): void {
		this.isEditModeActive = false;
		this.addFilesHandler = null;
	}

	isEditing(): boolean {
		return this.isEditModeActive;
	}

	setEditModeActive(handler: (files: File[]) => void): void {
		this.isEditModeActive = true;
		this.addFilesHandler = handler;
	}

	getAddFilesHandler(): ((files: File[]) => void) | null {
		return this.addFilesHandler;
	}

	clearPendingEditMessageId(): void {
		this.pendingEditMessageId = null;
	}

	savePendingDraft(message: string, files: ChatUploadedFile[]): void {
		this._pendingDraftMessage = message;
		this._pendingDraftFiles = [...files];
	}

	consumePendingDraft(): { message: string; files: ChatUploadedFile[] } | null {
		if (!this._pendingDraftMessage && this._pendingDraftFiles.length === 0) return null;
		const d = { message: this._pendingDraftMessage, files: [...this._pendingDraftFiles] };
		this._pendingDraftMessage = '';
		this._pendingDraftFiles = [];
		return d;
	}

	hasPendingDraft(): boolean {
		return Boolean(this._pendingDraftMessage) || this._pendingDraftFiles.length > 0;
	}

	getAllLoadingChats(): string[] {
		return Array.from(this.chatLoadingStates.keys());
	}

	getAllStreamingChats(): string[] {
		return Array.from(this.chatStreamingStates.keys());
	}

	getChatStreamingPublic(convId: string): { response: string; messageId: string } | undefined {
		return this.getChatStreaming(convId);
	}

	isChatLoadingPublic(convId: string): boolean {
		return this.chatLoadingStates.get(convId) || false;
	}

	isChatReasoningPublic(convId: string): boolean {
		return this.chatReasoningStates.get(convId) || false;
	}

	private isChatLoadingInternal(convId: string): boolean {
		return this.chatLoadingStates.has(convId) || this.chatStreamingStates.has(convId);
	}

	hasPendingMessage(convId: string): boolean {
		return this._pendingMessages.has(convId);
	}

	pendingMessageContent(convId: string): string | null {
		return this._pendingMessages.get(convId)?.content ?? null;
	}

	pendingMessageExtras(convId: string): DatabaseMessageExtra[] | undefined {
		return this._pendingMessages.get(convId)?.extras;
	}

	injectPendingMessage(convId: string, content: string, extras?: DatabaseMessageExtra[]): void {
		this._pendingMessages.set(convId, { content, extras });
	}

	clearPendingMessage(convId: string): void {
		this._pendingMessages.delete(convId);
	}

	consumePendingMessage(
		convId: string
	): { content: string; extras?: DatabaseMessageExtra[] } | null {
		const msg = this._pendingMessages.get(convId);
		if (!msg) return null;
		this._pendingMessages.delete(convId);
		return msg;
	}

	private touchConversationState(convId: string): void {
		this.conversationStateTimestamps.set(convId, { lastAccessed: Date.now() });
	}

	cleanupOldConversationStates(activeConversationIds?: string[]): number {
		const now = Date.now();
		const activeIdsList = activeConversationIds ?? [];
		const preserveIds = this.activeConversationId
			? [...activeIdsList, this.activeConversationId]
			: activeIdsList;
		const allConvIds = [
			...new Set([
				...this.chatLoadingStates.keys(),
				...this.chatStreamingStates.keys(),
				...this.abortControllers.keys(),
				...this.processingStates.keys(),
				...this.conversationStateTimestamps.keys()
			])
		];
		const cleanupCandidates: Array<{ convId: string; lastAccessed: number }> = [];
		for (const convId of allConvIds) {
			if (preserveIds.includes(convId)) continue;
			if (this.chatLoadingStates.get(convId)) continue;
			if (this.chatStreamingStates.has(convId)) continue;
			const ts = this.conversationStateTimestamps.get(convId);
			cleanupCandidates.push({ convId, lastAccessed: ts?.lastAccessed ?? 0 });
		}
		cleanupCandidates.sort((a, b) => a.lastAccessed - b.lastAccessed);
		let cleanedUp = 0;
		for (const { convId, lastAccessed } of cleanupCandidates) {
			if (
				cleanupCandidates.length - cleanedUp > MAX_INACTIVE_CONVERSATION_STATES ||
				now - lastAccessed > INACTIVE_CONVERSATION_STATE_MAX_AGE_MS
			) {
				this.cleanupConversationState(convId);
				cleanedUp++;
			}
		}
		return cleanedUp;
	}
	private cleanupConversationState(convId: string): void {
		const c = this.abortControllers.get(convId);
		if (c && !c.signal.aborted) c.abort();
		this.chatLoadingStates.delete(convId);
		this.chatStreamingStates.delete(convId);
		this.abortControllers.delete(convId);
		this.processingStates.delete(convId);
		this.conversationStateTimestamps.delete(convId);
	}
	getTrackedConversationCount(): number {
		return new Set([
			...this.chatLoadingStates.keys(),
			...this.chatStreamingStates.keys(),
			...this.abortControllers.keys(),
			...this.processingStates.keys()
		]).size;
	}

	private getMessageByIdWithRole(
		messageId: string,
		expectedRole?: MessageRole
	): { message: DatabaseMessage; index: number } | null {
		const index = conversationsStore.findMessageIndex(messageId);
		if (index === -1) return null;
		const message = conversationsStore.activeMessages[index];
		if (expectedRole && message.role !== expectedRole) return null;
		return { message, index };
	}

	async addMessage(
		role: MessageRole,
		content: string,
		type: MessageType = MessageType.TEXT,
		parent: string = '-1',
		extras?: DatabaseMessageExtra[]
	): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) throw new Error('No active conversation');
		let parentId: string | null = null;
		if (parent === '-1') {
			const am = conversationsStore.activeMessages;
			if (am.length > 0) parentId = am[am.length - 1].id;
			else {
				const all = await conversationsStore.getConversationMessages(activeConv.id);
				const r = all.find((m) => m.parent === null && m.type === 'root');
				parentId = r ? r.id : await DatabaseService.createRootMessage(activeConv.id);
			}
		} else parentId = parent;
		const message = await DatabaseService.createMessageBranch(
			{
				convId: activeConv.id,
				role,
				content,
				type,
				timestamp: Date.now(),
				toolCalls: '',
				children: [],
				extra: extras
			},
			parentId
		);
		conversationsStore.addMessageToActive(message);
		await conversationsStore.updateCurrentNode(message.id);
		conversationsStore.updateConversationTimestamp();
		return message;
	}

	async addSystemPrompt(): Promise<void> {
		let activeConv = conversationsStore.activeConversation;
		if (!activeConv) {
			await conversationsStore.createConversation();
			activeConv = conversationsStore.activeConversation;
		}
		if (!activeConv) return;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const rootId = rootMessage
				? rootMessage.id
				: await DatabaseService.createRootMessage(activeConv.id);
			const existingSystemMessage = allMessages.find(
				(m) => m.role === MessageRole.SYSTEM && m.parent === rootId
			);
			if (existingSystemMessage) {
				this.pendingEditMessageId = existingSystemMessage.id;
				if (!conversationsStore.activeMessages.some((m) => m.id === existingSystemMessage.id))
					conversationsStore.activeMessages.unshift(existingSystemMessage);
				return;
			}
			const am = conversationsStore.activeMessages;
			const firstActiveMessage = am.find((m) => m.parent === rootId);
			const systemMessage = await DatabaseService.createSystemMessage(
				activeConv.id,
				SYSTEM_MESSAGE_PLACEHOLDER,
				rootId
			);
			if (firstActiveMessage) {
				await DatabaseService.updateMessage(firstActiveMessage.id, {
					parent: systemMessage.id
				});
				await DatabaseService.updateMessage(systemMessage.id, {
					children: [firstActiveMessage.id]
				});
				const updatedRootChildren = rootMessage
					? rootMessage.children.filter((id: string) => id !== firstActiveMessage.id)
					: [];
				await DatabaseService.updateMessage(rootId, {
					children: [
						...updatedRootChildren.filter((id: string) => id !== systemMessage.id),
						systemMessage.id
					]
				});
				const firstMsgIndex = conversationsStore.findMessageIndex(firstActiveMessage.id);
				if (firstMsgIndex !== -1)
					conversationsStore.updateMessageAtIndex(firstMsgIndex, {
						parent: systemMessage.id
					});
			}
			conversationsStore.activeMessages.unshift(systemMessage);
			this.pendingEditMessageId = systemMessage.id;
			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to add system prompt:', error);
		}
	}

	async removeSystemPromptPlaceholder(messageId: string): Promise<boolean> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return false;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const systemMessage = findMessageById(allMessages, messageId);
			if (!systemMessage || systemMessage.role !== MessageRole.SYSTEM) return false;
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			if (!rootMessage) return false;
			if (allMessages.length === 2 && systemMessage.children.length === 0) {
				await conversationsStore.deleteConversation(activeConv.id);
				return true;
			}
			for (const childId of systemMessage.children) {
				await DatabaseService.updateMessage(childId, { parent: rootMessage.id });
				const childIndex = conversationsStore.findMessageIndex(childId);
				if (childIndex !== -1)
					conversationsStore.updateMessageAtIndex(childIndex, { parent: rootMessage.id });
			}
			await DatabaseService.updateMessage(rootMessage.id, {
				children: [
					...rootMessage.children.filter((id: string) => id !== messageId),
					...systemMessage.children
				]
			});
			await DatabaseService.deleteMessage(messageId);
			const systemIndex = conversationsStore.findMessageIndex(messageId);
			if (systemIndex !== -1) conversationsStore.activeMessages.splice(systemIndex, 1);
			conversationsStore.updateConversationTimestamp();
			return false;
		} catch (error) {
			console.error('Failed to remove system prompt placeholder:', error);
			return false;
		}
	}

	private async createAssistantMessage(parentId?: string): Promise<DatabaseMessage> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) throw new Error('No active conversation');
		return await DatabaseService.createMessageBranch(
			{
				convId: activeConv.id,
				type: MessageType.TEXT,
				role: MessageRole.ASSISTANT,
				content: '',
				timestamp: Date.now(),
				toolCalls: '',
				children: [],
				model: null
			},
			parentId || null
		);
	}

	async sendMessage(content: string, extras?: DatabaseMessageExtra[]): Promise<void> {
		if (!content.trim() && (!extras || extras.length === 0)) return;
		const activeConv = conversationsStore.activeConversation;

		// If agentic loop is running, inject as a steering message instead of starting a new flow
		if (activeConv && agenticStore.isRunning(activeConv.id)) {
			agenticStore.injectSteeringMessage(activeConv.id, content, extras);
			return;
		}

		// If non-agentic streaming is active, queue as a pending message to send after completion
		if (activeConv && this.isChatLoadingInternal(activeConv.id)) {
			this.injectPendingMessage(activeConv.id, content, extras);
			return;
		}

		// Cancel any in-flight pre-encode request
		this.cancelPreEncode();

		// Consume MCP resource attachments - converts them to extras and clears the live store
		const resourceExtras = mcpStore.consumeResourceAttachmentsAsExtras();
		const allExtras = resourceExtras.length > 0 ? [...(extras || []), ...resourceExtras] : extras;

		let isNewConversation = false;
		if (!activeConv) {
			await conversationsStore.createConversation();
			isNewConversation = true;
		}
		const currentConv = conversationsStore.activeConversation;
		if (!currentConv) return;
		this.showErrorDialog(null);
		this.setChatLoading(currentConv.id, true);
		this.clearChatStreaming(currentConv.id);
		try {
			let parentIdForUserMessage: string | undefined;
			if (isNewConversation) {
				const rootId = await DatabaseService.createRootMessage(currentConv.id);
				const currentConfig = config();
				const systemPrompt = currentConfig.systemMessage?.toString().trim();
				if (systemPrompt) {
					const systemMessage = await DatabaseService.createSystemMessage(
						currentConv.id,
						systemPrompt,
						rootId
					);
					conversationsStore.addMessageToActive(systemMessage);
					parentIdForUserMessage = systemMessage.id;
				} else parentIdForUserMessage = rootId;
			}
			const userMessage = await this.addMessage(
				MessageRole.USER,
				content,
				MessageType.TEXT,
				parentIdForUserMessage ?? '-1',
				allExtras
			);
			if (isNewConversation && content)
				await conversationsStore.updateConversationName(
					currentConv.id,
					generateConversationTitle(content, Boolean(config().titleGenerationUseFirstLine))
				);
			const assistantMessage = await this.createAssistantMessage(userMessage.id);
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage,
				undefined,
				undefined,
				undefined,
				config().titleGenerationUseLLM && isNewConversation ? content : undefined
			);
		} catch (error) {
			if (isAbortError(error)) {
				this.setChatLoading(currentConv.id, false);
				return;
			}
			console.error('Failed to send message:', error);
			this.setChatLoading(currentConv.id, false);
			const dialogType =
				error instanceof Error && error.name === 'TimeoutError'
					? ErrorDialogType.TIMEOUT
					: ErrorDialogType.SERVER;
			const contextInfo = (
				error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
			).contextInfo;
			this.showErrorDialog({
				type: dialogType,
				message: error instanceof Error ? error.message : 'Unknown error',
				contextInfo
			});
		}
	}

	private async streamChatCompletion(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		onComplete?: (content: string) => Promise<void>,
		onError?: (error: Error) => void,
		modelOverride?: string | null,
		firstUserMessageContent?: string
	): Promise<void> {
		let effectiveModel = modelOverride;

		if (isRouterMode() && !effectiveModel) {
			const conversationModel = this.getConversationModel(allMessages);
			effectiveModel = selectedModelName() || conversationModel;
		}

		if (isRouterMode() && effectiveModel) {
			if (!modelsStore.getModelProps(effectiveModel))
				await modelsStore.fetchModelProps(effectiveModel);
		}

		// Mutable state for the current message being streamed
		let currentMessageId = assistantMessage.id;
		let streamedContent = '';
		let streamedReasoningContent = '';
		let resolvedModel: string | null = null;
		let modelPersisted = false;
		const convId = assistantMessage.convId;

		const recordModel = (modelName: string | null | undefined, persistImmediately = true): void => {
			if (!modelName) return;
			const n = normalizeModelName(modelName);
			if (!n || n === resolvedModel) return;
			resolvedModel = n;
			const idx = conversationsStore.findMessageIndex(currentMessageId);
			conversationsStore.updateMessageAtIndex(idx, { model: n });
			if (persistImmediately && !modelPersisted) {
				modelPersisted = true;
				DatabaseService.updateMessage(currentMessageId, { model: n }).catch(() => {
					modelPersisted = false;
					resolvedModel = null;
				});
			}
		};

		let completionIdRecorded = false;
		const recordCompletionId = (id: string): void => {
			if (!id || completionIdRecorded) return;
			completionIdRecorded = true;
			const idx = conversationsStore.findMessageIndex(currentMessageId);
			conversationsStore.updateMessageAtIndex(idx, { completionId: id });
			DatabaseService.updateMessage(currentMessageId, { completionId: id }).catch(() => {
				completionIdRecorded = false;
			});
		};

		const updateStreamingUI = () => {
			this.setChatStreaming(convId, streamedContent, currentMessageId);
			const idx = conversationsStore.findMessageIndex(currentMessageId);
			conversationsStore.updateMessageAtIndex(idx, { content: streamedContent });
		};

		const cleanupStreamingState = () => {
			this.setStreamingActive(false);
			this.setChatLoading(convId, false);
			this.clearChatStreaming(convId);
			this.setProcessingState(convId, null);
		};

		this.setStreamingActive(true);
		this.setActiveProcessingConversation(convId);
		const abortController = this.getOrCreateAbortController(convId);

		const streamCallbacks: ChatStreamCallbacks = {
			onChunk: (chunk: string) => {
				streamedContent += chunk;
				updateStreamingUI();
				this.setChatReasoning(convId, false);
			},
			onReasoningChunk: (chunk: string) => {
				streamedReasoningContent += chunk;
				// mark streaming state so a stop mid-thinking can persist the partial reasoning
				this.setChatStreaming(convId, streamedContent, currentMessageId);
				const idx = conversationsStore.findMessageIndex(currentMessageId);
				conversationsStore.updateMessageAtIndex(idx, {
					reasoningContent: streamedReasoningContent
				});
				this.setChatReasoning(convId, true);
			},
			onToolCallsStreaming: (toolCalls) => {
				const idx = conversationsStore.findMessageIndex(currentMessageId);
				conversationsStore.updateMessageAtIndex(idx, {
					toolCalls: JSON.stringify(toolCalls)
				});
			},
			onAttachments: (messageId: string, extras: DatabaseMessageExtra[]) => {
				if (!extras.length) return;
				const idx = conversationsStore.findMessageIndex(messageId);
				if (idx === -1) return;
				const msg = conversationsStore.activeMessages[idx];
				const updatedExtras = [...(msg.extra || []), ...extras];
				conversationsStore.updateMessageAtIndex(idx, { extra: updatedExtras });
				DatabaseService.updateMessage(messageId, { extra: updatedExtras }).catch(console.error);
			},
			onModel: (modelName: string) => recordModel(modelName),
			onCompletionId: (id: string) => recordCompletionId(id),
			onTurnComplete: (intermediateTimings: ChatMessageTimings) => {
				// Update the first assistant message with cumulative agentic timings
				const idx = conversationsStore.findMessageIndex(assistantMessage.id);
				conversationsStore.updateMessageAtIndex(idx, { timings: intermediateTimings });
			},
			onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
				const tokensPerSecond =
					timings?.predicted_ms && timings?.predicted_n
						? (timings.predicted_n / timings.predicted_ms) * 1000
						: 0;
				this.updateProcessingStateFromTimings(
					{
						prompt_n: timings?.prompt_n || 0,
						prompt_ms: timings?.prompt_ms,
						predicted_n: timings?.predicted_n || 0,
						predicted_per_second: tokensPerSecond,
						cache_n: timings?.cache_n || 0,
						prompt_progress: promptProgress
					},
					convId
				);
			},
			onAssistantTurnComplete: async (
				content: string,
				reasoningContent: string | undefined,
				timings: ChatMessageTimings | undefined,
				toolCalls: import('$lib/types/api').ApiChatCompletionToolCall[] | undefined
			) => {
				const updateData: Record<string, unknown> = {
					content,
					reasoningContent: reasoningContent || undefined,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : '',
					timings
				};
				if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;
				await DatabaseService.updateMessage(currentMessageId, updateData);
				const idx = conversationsStore.findMessageIndex(currentMessageId);
				const uiUpdate: Partial<DatabaseMessage> = {
					content,
					reasoningContent: reasoningContent || undefined,
					toolCalls: toolCalls ? JSON.stringify(toolCalls) : ''
				};
				if (timings) uiUpdate.timings = timings;
				if (resolvedModel) uiUpdate.model = resolvedModel;
				conversationsStore.updateMessageAtIndex(idx, uiUpdate);
				await conversationsStore.updateCurrentNode(currentMessageId);
			},
			createToolResultMessage: async (
				toolCallId: string,
				content: string,
				extras?: DatabaseMessageExtra[]
			) => {
				const msg = await DatabaseService.createMessageBranch(
					{
						convId,
						type: MessageType.TEXT,
						role: MessageRole.TOOL,
						content,
						toolCallId,
						timestamp: Date.now(),
						toolCalls: '',
						children: [],
						extra: extras
					},
					currentMessageId
				);
				conversationsStore.addMessageToActive(msg);
				await conversationsStore.updateCurrentNode(msg.id);
				return msg;
			},
			createAssistantMessage: async () => {
				// Reset streaming state for new message
				streamedContent = '';
				streamedReasoningContent = '';

				const lastMsg =
					conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1];
				const msg = await DatabaseService.createMessageBranch(
					{
						convId,
						type: MessageType.TEXT,
						role: MessageRole.ASSISTANT,
						content: '',
						timestamp: Date.now(),
						toolCalls: '',
						children: [],
						model: resolvedModel
					},
					lastMsg.id
				);
				conversationsStore.addMessageToActive(msg);
				currentMessageId = msg.id;
				return msg;
			},
			onFlowComplete: (finalTimings?: ChatMessageTimings) => {
				if (finalTimings) {
					const idx = conversationsStore.findMessageIndex(assistantMessage.id);

					conversationsStore.updateMessageAtIndex(idx, { timings: finalTimings });
					DatabaseService.updateMessage(assistantMessage.id, {
						timings: finalTimings
					}).catch(console.error);
				}

				cleanupStreamingState();

				if (onComplete) onComplete(streamedContent);
				if (isRouterMode()) modelsStore.fetchRouterModels().catch(console.error);
				// Pre-encode conversation in KV cache for faster next turn
				if (config().preEncodeConversation) {
					this.triggerPreEncode(
						allMessages,
						assistantMessage,
						streamedContent,
						effectiveModel,
						!!config().excludeReasoningFromContext
					);
				}
			},
			onError: async (error: Error) => {
				this.setStreamingActive(false);
				if (isAbortError(error)) {
					cleanupStreamingState();
					// If aborted with a pending message (e.g. "Send immediately"), re-send it
					const pending = this.consumePendingMessage(convId);
					if (pending) {
						this.sendMessage(pending.content, pending.extras);
					}
					return;
				}
				console.error('Streaming error:', error);
				// keep whatever was streamed so far, the message stays in memory and in DB
				await this.savePartialResponseIfNeeded(convId);
				cleanupStreamingState();
				this.clearPendingMessage(convId);

				const contextInfo = (
					error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
				).contextInfo;
				this.showErrorDialog({
					type: error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
					message: error.message,
					contextInfo
				});
				if (onError) onError(error);
			}
		};

		const perChatOverrides = conversationsStore.activeConversation?.mcpServerOverrides;

		{
			const agenticResult = await agenticStore.runAgenticFlow({
				conversationId: convId,
				messages: allMessages,
				options: {
					...this.getApiOptions(),
					...(effectiveModel ? { model: effectiveModel } : {})
				},
				callbacks: streamCallbacks,
				signal: abortController.signal,
				perChatOverrides
			});
			if (agenticResult.handled) {
				// Generate LLM based title for new conversations after agentic flow completes
				if (firstUserMessageContent) {
					await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
				}
				// Check if there's a pending steering message to re-send
				const pending = agenticStore.consumePendingSteeringMessage(convId);
				if (pending) {
					await this.sendMessage(pending.content, pending.extras);
				}
				return;
			}
		}

		await ChatService.sendMessage(
			allMessages,
			{
				...this.getApiOptions(),
				...(effectiveModel ? { model: effectiveModel } : {}),
				stream: true,
				onChunk: streamCallbacks.onChunk,
				onReasoningChunk: streamCallbacks.onReasoningChunk,
				onModel: streamCallbacks.onModel,
				onCompletionId: streamCallbacks.onCompletionId,
				onTimings: streamCallbacks.onTimings,
				onComplete: async (
					finalContent?: string,
					reasoningContent?: string,
					timings?: ChatMessageTimings,
					toolCalls?: string
				) => {
					const content = streamedContent || finalContent || '';
					const reasoning = streamedReasoningContent || reasoningContent;
					const updateData: Record<string, unknown> = {
						content,
						reasoningContent: reasoning || undefined,
						toolCalls: toolCalls || '',
						timings
					};
					if (resolvedModel && !modelPersisted) updateData.model = resolvedModel;
					await DatabaseService.updateMessage(currentMessageId, updateData);
					const idx = conversationsStore.findMessageIndex(currentMessageId);
					const uiUpdate: Partial<DatabaseMessage> = {
						content,
						reasoningContent: reasoning || undefined,
						toolCalls: toolCalls || ''
					};
					if (timings) uiUpdate.timings = timings;
					if (resolvedModel) uiUpdate.model = resolvedModel;
					conversationsStore.updateMessageAtIndex(idx, uiUpdate);
					await conversationsStore.updateCurrentNode(currentMessageId);
					cleanupStreamingState();
					if (onComplete) await onComplete(content);
					if (isRouterMode()) modelsStore.fetchRouterModels().catch(console.error);

					// Generate LLM based title for new conversations (avoids stale reference
					// issue when user switches conversations while streaming)
					if (firstUserMessageContent) {
						await this.generateTitleWithLLM(firstUserMessageContent, streamedContent, convId);
					}

					// Check if there's a pending message queued during streaming
					const pending = this.consumePendingMessage(convId);
					if (pending) {
						await this.sendMessage(pending.content, pending.extras);
					}
				},
				onError: streamCallbacks.onError
			},
			convId,
			abortController.signal
		);
	}

	async stopGeneration(): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		await this.stopGenerationForChat(activeConv.id);
	}
	async stopGenerationForChat(convId: string): Promise<void> {
		await this.savePartialResponseIfNeeded(convId);
		this.setStreamingActive(false);
		this.abortRequest(convId);
		this.setChatLoading(convId, false);
		this.clearChatStreaming(convId);
		this.setProcessingState(convId, null);
		this.clearPendingMessage(convId);
	}

	private async generateTitleWithLLM(
		userContent: string,
		assistantContent: string,
		convId: string
	): Promise<void> {
		const effectiveModel = isRouterMode() && selectedModelName() ? selectedModelName() : undefined;
		const configValue = config();
		const titlePromptTemplate =
			typeof configValue.titleGenerationPrompt === 'string' &&
			configValue.titleGenerationPrompt.trim()
				? configValue.titleGenerationPrompt
				: TITLE_GENERATION.DEFAULT_PROMPT;

		const titlePrompt = titlePromptTemplate
			.replace('{{USER}}', String(userContent || ''))
			.replace('{{ASSISTANT}}', String(assistantContent || ''));

		const titleMessage: ApiChatMessageData = {
			role: MessageRole.USER,
			content: titlePrompt
		};

		const titleResponse = await ChatService.generateTitle(titleMessage, effectiveModel);

		if (!titleResponse) {
			return;
		}

		let cleanTitle = titleResponse.trim();
		cleanTitle = cleanTitle
			.replace(TITLE_GENERATION.PREFIX_PATTERN, '')
			.replace(TITLE_GENERATION.QUOTE_PATTERN, '')
			.trim();
		if (!cleanTitle || cleanTitle.length < TITLE_GENERATION.MIN_LENGTH) {
			const firstLine = userContent.split('\n').find((l) => l.trim().length > 0);
			cleanTitle = firstLine ? firstLine.trim() : TITLE_GENERATION.FALLBACK;
		}
		if (cleanTitle && cleanTitle.length >= TITLE_GENERATION.MIN_LENGTH) {
			await conversationsStore.updateConversationName(convId, cleanTitle);
		}
	}

	private async savePartialResponseIfNeeded(convId?: string): Promise<void> {
		const conversationId = convId || conversationsStore.activeConversation?.id;
		if (!conversationId) return;
		const streamingState = this.getChatStreaming(conversationId);
		if (!streamingState) return;
		const messages =
			conversationId === conversationsStore.activeConversation?.id
				? conversationsStore.activeMessages
				: await conversationsStore.getConversationMessages(conversationId);
		if (!messages.length) return;
		const lastMessage = messages[messages.length - 1];
		if (lastMessage?.role !== MessageRole.ASSISTANT) return;

		const partialContent = streamingState.response;
		const partialReasoning = lastMessage.reasoningContent || '';

		// nothing to persist when both content and reasoning are empty (e.g. stop before any token)
		if (!partialContent.trim() && !partialReasoning.trim()) return;

		try {
			const updateData: {
				content: string;
				reasoningContent?: string;
				timings?: ChatMessageTimings;
			} = {
				content: partialContent
			};
			if (partialReasoning) {
				updateData.reasoningContent = partialReasoning;
			}
			const lastKnownState = this.getProcessingState(conversationId);
			if (lastKnownState) {
				updateData.timings = {
					prompt_n: lastKnownState.promptTokens || 0,
					prompt_ms: lastKnownState.promptMs,
					predicted_n: lastKnownState.tokensDecoded || 0,
					cache_n: lastKnownState.cacheTokens || 0,
					predicted_ms:
						lastKnownState.tokensPerSecond && lastKnownState.tokensDecoded
							? (lastKnownState.tokensDecoded / lastKnownState.tokensPerSecond) * 1000
							: undefined
				};
			}
			await DatabaseService.updateMessage(lastMessage.id, updateData);
			lastMessage.content = partialContent;
			if (updateData.timings) lastMessage.timings = updateData.timings;
		} catch (error) {
			lastMessage.content = partialContent;
			console.error('Failed to save partial response:', error);
		}
	}

	async updateMessage(messageId: string, newContent: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		if (this.isChatLoadingInternal(activeConv.id)) await this.stopGeneration();
		const result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) return;
		const { message: messageToUpdate, index: messageIndex } = result;
		const originalContent = messageToUpdate.content;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage = rootMessage && messageToUpdate.parent === rootMessage.id;
			conversationsStore.updateMessageAtIndex(messageIndex, { content: newContent });
			await DatabaseService.updateMessage(messageId, { content: newContent });
			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					generateConversationTitle(newContent, Boolean(config().titleGenerationUseFirstLine))
				);
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex + 1);
			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);
			conversationsStore.sliceActiveMessages(messageIndex + 1);
			conversationsStore.updateConversationTimestamp();
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);
			const assistantMessage = await this.createAssistantMessage();
			conversationsStore.addMessageToActive(assistantMessage);
			await conversationsStore.updateCurrentNode(assistantMessage.id);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage,
				undefined,
				() => {
					conversationsStore.updateMessageAtIndex(conversationsStore.findMessageIndex(messageId), {
						content: originalContent
					});
				}
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to update message:', error);
		}
	}

	async regenerateMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		this.cancelPreEncode();
		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);
		if (!result) return;
		const { index: messageIndex } = result;
		try {
			const messagesToRemove = conversationsStore.activeMessages.slice(messageIndex);
			for (const message of messagesToRemove) await DatabaseService.deleteMessage(message.id);
			conversationsStore.sliceActiveMessages(messageIndex);
			conversationsStore.updateConversationTimestamp();
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);
			const parentMessageId =
				conversationsStore.activeMessages.length > 0
					? conversationsStore.activeMessages[conversationsStore.activeMessages.length - 1].id
					: undefined;
			const assistantMessage = await this.createAssistantMessage(parentMessageId);
			conversationsStore.addMessageToActive(assistantMessage);
			await this.streamChatCompletion(
				conversationsStore.activeMessages.slice(0, -1),
				assistantMessage
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to regenerate message:', error);
			this.setChatLoading(activeConv?.id || '', false);
		}
	}

	async regenerateMessageWithBranching(messageId: string, modelOverride?: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		this.cancelPreEncode();
		try {
			const idx = conversationsStore.findMessageIndex(messageId);
			if (idx === -1) return;
			const msg = conversationsStore.activeMessages[idx];
			if (msg.role !== MessageRole.ASSISTANT) return;
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const parentMessage = findMessageById(allMessages, msg.parent);
			if (!parentMessage) return;
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);
			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: msg.convId,
					type: msg.type,
					timestamp: Date.now(),
					role: msg.role,
					content: '',
					toolCalls: '',
					children: [],
					model: null
				},
				parentMessage.id
			);
			await conversationsStore.updateCurrentNode(newAssistantMessage.id);
			conversationsStore.updateConversationTimestamp();
			await conversationsStore.refreshActiveMessages();
			const conversationPath = filterByLeafNodeId(
				allMessages,
				parentMessage.id,
				false
			) as DatabaseMessage[];
			const modelToUse = modelOverride || msg.model || undefined;
			await this.streamChatCompletion(
				conversationPath,
				newAssistantMessage,
				undefined,
				undefined,
				modelToUse
			);
		} catch (error) {
			if (!isAbortError(error))
				console.error('Failed to regenerate message with branching:', error);
			this.setChatLoading(activeConv?.id || '', false);
		}
	}

	async getDeletionInfo(messageId: string): Promise<{
		totalCount: number;
		userMessages: number;
		assistantMessages: number;
		messageTypes: string[];
	}> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv)
			return { totalCount: 0, userMessages: 0, assistantMessages: 0, messageTypes: [] };
		const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
		const messageToDelete = findMessageById(allMessages, messageId);

		// For system messages, don't count descendants as they will be preserved (reparented to root)
		if (messageToDelete?.role === MessageRole.SYSTEM) {
			const messagesToDelete = allMessages.filter((m) => m.id === messageId);
			let userMessages = 0,
				assistantMessages = 0;
			const messageTypes: string[] = [];

			for (const msg of messagesToDelete) {
				if (msg.role === MessageRole.USER) {
					userMessages++;
					if (!messageTypes.includes('user message')) messageTypes.push('user message');
				} else if (msg.role === MessageRole.ASSISTANT) {
					assistantMessages++;
					if (!messageTypes.includes('assistant response')) messageTypes.push('assistant response');
				}
			}

			return { totalCount: 1, userMessages, assistantMessages, messageTypes };
		}

		const descendants = findDescendantMessages(allMessages, messageId);
		const allToDelete = [messageId, ...descendants];
		const messagesToDelete = allMessages.filter((m) => allToDelete.includes(m.id));
		let userMessages = 0,
			assistantMessages = 0;
		const messageTypes: string[] = [];

		for (const msg of messagesToDelete) {
			if (msg.role === MessageRole.USER) {
				userMessages++;
				if (!messageTypes.includes('user message')) messageTypes.push('user message');
			} else if (msg.role === MessageRole.ASSISTANT) {
				assistantMessages++;
				if (!messageTypes.includes('assistant response')) messageTypes.push('assistant response');
			}
		}

		return { totalCount: allToDelete.length, userMessages, assistantMessages, messageTypes };
	}

	async deleteMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const messageToDelete = findMessageById(allMessages, messageId);

			if (!messageToDelete) return;

			const currentPath = filterByLeafNodeId(allMessages, activeConv.currNode || '', false);
			const isInCurrentPath = currentPath.some((m) => m.id === messageId);

			if (isInCurrentPath && messageToDelete.parent) {
				const siblings = allMessages.filter(
					(m) => m.parent === messageToDelete.parent && m.id !== messageId
				);

				if (siblings.length > 0) {
					const latestSibling = siblings.reduce((latest, sibling) =>
						sibling.timestamp > latest.timestamp ? sibling : latest
					);

					await conversationsStore.updateCurrentNode(findLeafNode(allMessages, latestSibling.id));
				} else if (messageToDelete.parent) {
					await conversationsStore.updateCurrentNode(
						findLeafNode(allMessages, messageToDelete.parent)
					);
				}
			}

			await DatabaseService.deleteMessageCascading(activeConv.id, messageId);
			await conversationsStore.refreshActiveMessages();

			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to delete message:', error);
		}
	}

	/**
	 * Open a fresh assistant turn anchored at the last tool result of a resolved
	 * agentic round and let streamChatCompletion route through runAgenticFlow.
	 * Used by continueAssistantMessage when classifyContinueIntent returns
	 * next_turn, meaning the target assistant already has its tool_calls paired
	 * with trailing tool results and the next thing to generate is a brand new
	 * turn rather than a token level continuation.
	 */
	private async continueAsNextAgenticTurn(anchorIndex: number): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;
		const anchor = conversationsStore.activeMessages[anchorIndex];
		if (!anchor) return;
		this.cancelPreEncode();
		this.setChatLoading(activeConv.id, true);
		this.clearChatStreaming(activeConv.id);
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const anchorMessage = findMessageById(allMessages, anchor.id);
			if (!anchorMessage) {
				this.setChatLoading(activeConv.id, false);
				return;
			}
			const newAssistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: activeConv.id,
					type: MessageType.TEXT,
					timestamp: Date.now(),
					role: MessageRole.ASSISTANT,
					content: '',
					toolCalls: '',
					children: [],
					model: null
				},
				anchorMessage.id
			);
			await conversationsStore.updateCurrentNode(newAssistantMessage.id);
			conversationsStore.updateConversationTimestamp();
			await conversationsStore.refreshActiveMessages();
			const conversationPath = filterByLeafNodeId(
				allMessages,
				anchorMessage.id,
				false
			) as DatabaseMessage[];
			await this.streamChatCompletion(conversationPath, newAssistantMessage);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to continue agentic turn:', error);
			this.setChatLoading(activeConv.id, false);
		}
	}

	async continueAssistantMessage(messageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);

		if (!result) return;

		const { message: msg, index: idx } = result;

		// Decide which resume path applies. tool_calls without tool results can
		// not be resumed mid sequence by continue_final_message, branch instead.
		// tool_calls already paired with tool results need a fresh next turn,
		// not a token level continuation of the target assistant.
		const intent = classifyContinueIntent(conversationsStore.activeMessages, idx);
		if (intent.kind === ContinueIntentKind.RERUN_TURN) {
			return this.regenerateMessageWithBranching(messageId);
		}
		if (intent.kind === ContinueIntentKind.NEXT_TURN) {
			return this.continueAsNextAgenticTurn(intent.truncateAfter);
		}

		try {
			this.showErrorDialog(null);
			this.setChatLoading(activeConv.id, true);
			this.clearChatStreaming(activeConv.id);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const dbMessage = findMessageById(allMessages, messageId);

			if (!dbMessage) {
				this.setChatLoading(activeConv.id, false);
				return;
			}

			const originalContent = dbMessage.content;
			const originalReasoning = dbMessage.reasoningContent || '';
			// Hand the persisted DatabaseMessage straight to sendMessage so its
			// internal converter preserves tool_calls and extras when present.
			// Reconstructing a bare {role, content} here would drop those fields
			// and break continue_final_message for messages with tool calls.
			const contextWithContinue = conversationsStore.activeMessages.slice(0, idx + 1);

			let appendedContent = '';
			let appendedReasoning = '';
			let hasReceivedContent = false;

			const updateStreamingContent = (fullContent: string) => {
				this.setChatStreaming(msg.convId, fullContent, msg.id);
				conversationsStore.updateMessageAtIndex(idx, { content: fullContent });
			};

			const abortController = this.getOrCreateAbortController(msg.convId);

			await ChatService.sendMessage(
				contextWithContinue,
				{
					...this.getApiOptions(),
					continueFinalMessage: true,
					onChunk: (chunk: string) => {
						appendedContent += chunk;
						hasReceivedContent = true;
						updateStreamingContent(originalContent + appendedContent);
						this.setChatReasoning(msg.convId, false);
					},
					onReasoningChunk: (chunk: string) => {
						appendedReasoning += chunk;
						hasReceivedContent = true;
						// mark streaming state so a stop mid-thinking can persist the partial reasoning
						this.setChatStreaming(msg.convId, originalContent + appendedContent, msg.id);
						conversationsStore.updateMessageAtIndex(idx, {
							reasoningContent: originalReasoning + appendedReasoning
						});
						this.setChatReasoning(msg.convId, true);
					},
					onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
						const tokensPerSecond =
							timings?.predicted_ms && timings?.predicted_n
								? (timings.predicted_n / timings.predicted_ms) * 1000
								: 0;
						this.updateProcessingStateFromTimings(
							{
								prompt_n: timings?.prompt_n || 0,
								prompt_ms: timings?.prompt_ms,
								predicted_n: timings?.predicted_n || 0,
								predicted_per_second: tokensPerSecond,
								cache_n: timings?.cache_n || 0,
								prompt_progress: promptProgress
							},
							msg.convId
						);
					},
					onComplete: async (
						finalContent?: string,
						reasoningContent?: string,
						timings?: ChatMessageTimings
					) => {
						const finalAppendedContent = hasReceivedContent ? appendedContent : finalContent || '';
						const finalAppendedReasoning = hasReceivedContent
							? appendedReasoning
							: reasoningContent || '';
						const fullContent = originalContent + finalAppendedContent;
						const fullReasoning = originalReasoning + finalAppendedReasoning || undefined;

						await DatabaseService.updateMessage(msg.id, {
							content: fullContent,
							reasoningContent: fullReasoning,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateMessageAtIndex(idx, {
							content: fullContent,
							reasoningContent: fullReasoning,
							timestamp: Date.now(),
							timings
						});

						conversationsStore.updateConversationTimestamp();

						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.setProcessingState(msg.convId, null);
					},
					onError: async (error: Error) => {
						if (isAbortError(error)) {
							if (hasReceivedContent && appendedContent) {
								await DatabaseService.updateMessage(msg.id, {
									content: originalContent + appendedContent,
									reasoningContent: originalReasoning + appendedReasoning || undefined,
									timestamp: Date.now()
								});

								conversationsStore.updateMessageAtIndex(idx, {
									content: originalContent + appendedContent,
									reasoningContent: originalReasoning + appendedReasoning || undefined,
									timestamp: Date.now()
								});
							}

							this.setChatLoading(msg.convId, false);
							this.clearChatStreaming(msg.convId);
							this.setProcessingState(msg.convId, null);

							return;
						}

						console.error('Continue generation error:', error);
						// keep whatever was appended so far, the message stays in memory and in DB
						await DatabaseService.updateMessage(msg.id, {
							content: originalContent + appendedContent,
							reasoningContent: originalReasoning + appendedReasoning || undefined,
							timestamp: Date.now()
						});
						conversationsStore.updateMessageAtIndex(idx, {
							content: originalContent + appendedContent,
							reasoningContent: originalReasoning + appendedReasoning || undefined,
							timestamp: Date.now()
						});

						this.setChatLoading(msg.convId, false);
						this.clearChatStreaming(msg.convId);
						this.setProcessingState(msg.convId, null);
						this.showErrorDialog({
							type:
								error.name === 'TimeoutError' ? ErrorDialogType.TIMEOUT : ErrorDialogType.SERVER,
							message: error.message
						});
					}
				},

				msg.convId,
				abortController.signal
			);
		} catch (error) {
			if (!isAbortError(error)) console.error('Failed to continue message:', error);
			if (activeConv) this.setChatLoading(activeConv.id, false);
		}
	}

	async editAssistantMessage(
		messageId: string,
		newContent: string,
		shouldBranch: boolean
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;

		const result = this.getMessageByIdWithRole(messageId, MessageRole.ASSISTANT);
		if (!result) return;

		const { message: msg, index: idx } = result;

		try {
			if (shouldBranch) {
				const newMessage = await DatabaseService.createMessageBranch(
					{
						convId: msg.convId,
						type: msg.type,
						timestamp: Date.now(),
						role: msg.role,
						content: newContent,
						toolCalls: msg.toolCalls || '',
						children: [],
						model: msg.model
					},
					msg.parent!
				);

				await conversationsStore.updateCurrentNode(newMessage.id);
			} else {
				await DatabaseService.updateMessage(msg.id, { content: newContent });
				conversationsStore.updateMessageAtIndex(idx, { content: newContent });
			}

			conversationsStore.updateConversationTimestamp();

			await conversationsStore.refreshActiveMessages();
		} catch (error) {
			console.error('Failed to edit assistant message:', error);
		}
	}

	async editUserMessagePreserveResponses(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;

		const result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) return;

		const { message: msg, index: idx } = result;
		try {
			const updateData: Partial<DatabaseMessage> = { content: newContent };

			if (newExtras !== undefined) updateData.extra = JSON.parse(JSON.stringify(newExtras));

			await DatabaseService.updateMessage(messageId, updateData);

			conversationsStore.updateMessageAtIndex(idx, updateData);

			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);

			if (rootMessage && msg.parent === rootMessage.id && newContent.trim()) {
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					generateConversationTitle(newContent, Boolean(config().titleGenerationUseFirstLine))
				);
			}

			conversationsStore.updateConversationTimestamp();
		} catch (error) {
			console.error('Failed to edit user message:', error);
		}
	}

	async editMessageWithBranching(
		messageId: string,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv || this.isChatLoadingInternal(activeConv.id)) return;
		let result = this.getMessageByIdWithRole(messageId, MessageRole.USER);
		if (!result) result = this.getMessageByIdWithRole(messageId, MessageRole.SYSTEM);
		if (!result) return;
		const { message: msg, index: idx } = result;
		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const rootMessage = allMessages.find((m) => m.type === 'root' && m.parent === null);
			const isFirstUserMessage =
				msg.role === MessageRole.USER && rootMessage && msg.parent === rootMessage.id;
			const extrasToUse =
				newExtras !== undefined
					? JSON.parse(JSON.stringify(newExtras))
					: msg.extra
						? JSON.parse(JSON.stringify(msg.extra))
						: undefined;

			let messageIdForResponse: string;

			const dbMsg = findMessageById(allMessages, msg.id);
			const hasChildren = dbMsg ? dbMsg.children.length > 0 : msg.children.length > 0;

			if (!hasChildren) {
				// No responses after this message — update in place instead of branching
				const updates: Partial<DatabaseMessage> = {
					content: newContent,
					timestamp: Date.now(),
					extra: extrasToUse
				};
				await DatabaseService.updateMessage(msg.id, updates);
				conversationsStore.updateMessageAtIndex(idx, updates);
				messageIdForResponse = msg.id;
			} else {
				// Has children — create a new branch as sibling
				const parentId = msg.parent || rootMessage?.id;
				if (!parentId) return;
				const newMessage = await DatabaseService.createMessageBranch(
					{
						convId: msg.convId,
						type: msg.type,
						timestamp: Date.now(),
						role: msg.role,
						content: newContent,
						toolCalls: msg.toolCalls || '',
						children: [],
						extra: extrasToUse,
						model: msg.model
					},
					parentId
				);
				await conversationsStore.updateCurrentNode(newMessage.id);
				messageIdForResponse = newMessage.id;
			}

			conversationsStore.updateConversationTimestamp();
			if (isFirstUserMessage && newContent.trim())
				await conversationsStore.updateConversationTitleWithConfirmation(
					activeConv.id,
					generateConversationTitle(newContent, Boolean(config().titleGenerationUseFirstLine))
				);
			await conversationsStore.refreshActiveMessages();
			if (msg.role === MessageRole.USER)
				await this.generateResponseForMessage(messageIdForResponse);
		} catch (error) {
			console.error('Failed to edit message with branching:', error);
		}
	}

	private async generateResponseForMessage(userMessageId: string): Promise<void> {
		const activeConv = conversationsStore.activeConversation;
		if (!activeConv) return;

		this.showErrorDialog(null);
		this.setChatLoading(activeConv.id, true);
		this.clearChatStreaming(activeConv.id);

		try {
			const allMessages = await conversationsStore.getConversationMessages(activeConv.id);
			const conversationPath = filterByLeafNodeId(
				allMessages,
				userMessageId,
				false
			) as DatabaseMessage[];
			const assistantMessage = await DatabaseService.createMessageBranch(
				{
					convId: activeConv.id,
					type: MessageType.TEXT,
					timestamp: Date.now(),
					role: MessageRole.ASSISTANT,
					content: '',
					toolCalls: '',
					children: [],
					model: null
				},
				userMessageId
			);

			conversationsStore.addMessageToActive(assistantMessage);

			await this.streamChatCompletion(conversationPath, assistantMessage);
		} catch (error) {
			console.error('Failed to generate response:', error);
			this.setChatLoading(activeConv.id, false);
		}
	}

	private getContextTotal(): number | null {
		const activeConvId = this.activeConversationId;
		const activeState = activeConvId ? this.getProcessingState(activeConvId) : null;

		if (activeState && typeof activeState.contextTotal === 'number' && activeState.contextTotal > 0)
			return activeState.contextTotal;

		if (isRouterMode()) {
			const modelContextSize = selectedModelContextSize();

			if (typeof modelContextSize === 'number' && modelContextSize > 0) {
				return modelContextSize;
			}
		} else {
			const propsContextSize = contextSize();

			if (typeof propsContextSize === 'number' && propsContextSize > 0) {
				return propsContextSize;
			}
		}

		return null;
	}

	updateProcessingStateFromTimings(
		timingData: {
			prompt_n: number;
			prompt_ms?: number;
			predicted_n: number;
			predicted_per_second: number;
			cache_n: number;
			prompt_progress?: ChatMessagePromptProgress;
		},
		conversationId?: string
	): void {
		const processingState = this.parseTimingData(timingData);

		if (processingState === null) {
			console.warn('Failed to parse timing data - skipping update');
			return;
		}

		const targetId = conversationId || this.activeConversationId;
		if (targetId) {
			this.setProcessingState(targetId, processingState);
		}
	}

	private parseTimingData(timingData: Record<string, unknown>): ApiProcessingState | null {
		const promptTokens = (timingData.prompt_n as number) || 0,
			promptMs = (timingData.prompt_ms as number) || undefined,
			predictedTokens = (timingData.predicted_n as number) || 0,
			tokensPerSecond = (timingData.predicted_per_second as number) || 0,
			cacheTokens = (timingData.cache_n as number) || 0;
		const promptProgress = timingData.prompt_progress as
			| { total: number; cache: number; processed: number; time_ms: number }
			| undefined;
		const contextTotal = this.getContextTotal();
		const currentConfig = config();
		const outputTokensMax = currentConfig.max_tokens || -1;
		const contextUsed = promptTokens + cacheTokens + predictedTokens,
			outputTokensUsed = predictedTokens;
		const progressCache = promptProgress?.cache || 0,
			progressActualDone = (promptProgress?.processed ?? 0) - progressCache,
			progressActualTotal = (promptProgress?.total ?? 0) - progressCache;
		const progressPercent = promptProgress
			? Math.round((progressActualDone / progressActualTotal) * 100)
			: undefined;
		return {
			status: predictedTokens > 0 ? 'generating' : promptProgress ? 'preparing' : 'idle',
			tokensDecoded: predictedTokens,
			tokensRemaining: outputTokensMax - predictedTokens,
			contextUsed,
			contextTotal,
			outputTokensUsed,
			outputTokensMax,
			hasNextToken: predictedTokens > 0,
			tokensPerSecond,
			temperature: currentConfig.temperature ?? 0.8,
			topP: currentConfig.top_p ?? 0.95,
			speculative: false,
			progressPercent,
			promptProgress,
			promptTokens,
			promptMs,
			cacheTokens
		};
	}

	restoreProcessingStateFromMessages(messages: DatabaseMessage[], conversationId: string): void {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === MessageRole.ASSISTANT && message.timings) {
				const restoredState = this.parseTimingData({
					prompt_n: message.timings.prompt_n || 0,
					prompt_ms: message.timings.prompt_ms,
					predicted_n: message.timings.predicted_n || 0,
					predicted_per_second:
						message.timings.predicted_n && message.timings.predicted_ms
							? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
							: 0,
					cache_n: message.timings.cache_n || 0
				});
				if (restoredState) {
					this.setProcessingState(conversationId, restoredState);
					return;
				}
			}
		}
	}

	getConversationModel(messages: DatabaseMessage[]): string | null {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role === MessageRole.ASSISTANT && message.model) return message.model;
		}
		return null;
	}

	private getApiOptions(): Record<string, unknown> {
		const currentConfig = config();
		const hasValue = (value: unknown): boolean =>
			value !== undefined && value !== null && value !== '';
		const apiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

		if (isRouterMode()) {
			const modelName = selectedModelName();
			if (modelName) apiOptions.model = modelName;
		}

		if (currentConfig.systemMessage) apiOptions.systemMessage = currentConfig.systemMessage;

		if (currentConfig.disableReasoningParsing) apiOptions.disableReasoningParsing = true;

		if (currentConfig.excludeReasoningFromContext) apiOptions.excludeReasoningFromContext = true;

		apiOptions.enableThinking = conversationsStore.getThinkingEnabled();
		apiOptions.reasoningEffort = conversationsStore.getReasoningEffort();

		if (hasValue(currentConfig.temperature))
			apiOptions.temperature = Number(currentConfig.temperature);

		if (hasValue(currentConfig.max_tokens))
			apiOptions.max_tokens = Number(currentConfig.max_tokens);

		if (hasValue(currentConfig.dynatemp_range))
			apiOptions.dynatemp_range = Number(currentConfig.dynatemp_range);

		if (hasValue(currentConfig.dynatemp_exponent))
			apiOptions.dynatemp_exponent = Number(currentConfig.dynatemp_exponent);

		if (hasValue(currentConfig.top_k)) apiOptions.top_k = Number(currentConfig.top_k);

		if (hasValue(currentConfig.top_p)) apiOptions.top_p = Number(currentConfig.top_p);

		if (hasValue(currentConfig.min_p)) apiOptions.min_p = Number(currentConfig.min_p);

		if (hasValue(currentConfig.xtc_probability))
			apiOptions.xtc_probability = Number(currentConfig.xtc_probability);

		if (hasValue(currentConfig.xtc_threshold))
			apiOptions.xtc_threshold = Number(currentConfig.xtc_threshold);

		if (hasValue(currentConfig.typ_p)) apiOptions.typ_p = Number(currentConfig.typ_p);

		if (hasValue(currentConfig.repeat_last_n))
			apiOptions.repeat_last_n = Number(currentConfig.repeat_last_n);

		if (hasValue(currentConfig.repeat_penalty))
			apiOptions.repeat_penalty = Number(currentConfig.repeat_penalty);

		if (hasValue(currentConfig.presence_penalty))
			apiOptions.presence_penalty = Number(currentConfig.presence_penalty);

		if (hasValue(currentConfig.frequency_penalty))
			apiOptions.frequency_penalty = Number(currentConfig.frequency_penalty);

		if (hasValue(currentConfig.dry_multiplier))
			apiOptions.dry_multiplier = Number(currentConfig.dry_multiplier);

		if (hasValue(currentConfig.dry_base)) apiOptions.dry_base = Number(currentConfig.dry_base);

		if (hasValue(currentConfig.dry_allowed_length))
			apiOptions.dry_allowed_length = Number(currentConfig.dry_allowed_length);

		if (hasValue(currentConfig.dry_penalty_last_n))
			apiOptions.dry_penalty_last_n = Number(currentConfig.dry_penalty_last_n);

		if (currentConfig.samplers) apiOptions.samplers = currentConfig.samplers;

		apiOptions.backend_sampling = currentConfig.backend_sampling;

		if (currentConfig.customJson) apiOptions.custom = currentConfig.customJson;

		return apiOptions;
	}

	private cancelPreEncode(): void {
		if (this.preEncodeAbortController) {
			this.preEncodeAbortController.abort();
			this.preEncodeAbortController = null;
		}
	}

	private async triggerPreEncode(
		allMessages: DatabaseMessage[],
		assistantMessage: DatabaseMessage,
		assistantContent: string,
		model?: string | null,
		excludeReasoning?: boolean
	): Promise<void> {
		this.cancelPreEncode();
		this.preEncodeAbortController = new AbortController();

		const signal = this.preEncodeAbortController.signal;

		try {
			const allIdle = await ChatService.areAllSlotsIdle(model, signal);
			if (!allIdle || signal.aborted) return;

			const messagesWithAssistant: DatabaseMessage[] = [
				...allMessages,
				{ ...assistantMessage, content: assistantContent }
			];

			await ChatService.preEncode(messagesWithAssistant, model, excludeReasoning, signal);
		} catch (err) {
			if (!isAbortError(err)) {
				console.warn('[ChatStore] Pre-encode failed:', err);
			}
		}
	}
}

export const chatStore = new ChatStore();

export const activeProcessingState = () => chatStore.activeProcessingState;
export const currentResponse = () => chatStore.currentResponse;
export const errorDialog = () => chatStore.errorDialogState;
export const getAddFilesHandler = () => chatStore.getAddFilesHandler();
export const getAllLoadingChats = () => chatStore.getAllLoadingChats();
export const getAllStreamingChats = () => chatStore.getAllStreamingChats();
export const getChatStreaming = (convId: string) => chatStore.getChatStreamingPublic(convId);
export const isChatLoading = (convId: string) => chatStore.isChatLoadingPublic(convId);
export const isChatStreaming = () => chatStore.isStreaming();
export const isEditing = () => chatStore.isEditing();
export const isLoading = () => chatStore.isLoading;
export const isReasoning = () => chatStore.isReasoning;
export const pendingEditMessageId = () => chatStore.pendingEditMessageId;
export const chatHasPendingMessage = (convId: string) => chatStore.hasPendingMessage(convId);
export const chatPendingMessageContent = (convId: string) =>
	chatStore.pendingMessageContent(convId);
export const chatPendingMessageExtras = (convId: string) => chatStore.pendingMessageExtras(convId);
export const chatClearPendingMessage = (convId: string) => chatStore.clearPendingMessage(convId);
export const chatInjectPendingMessage = (
	convId: string,
	content: string,
	extras?: DatabaseMessageExtra[]
) => chatStore.injectPendingMessage(convId, content, extras);
>>>>>>> Stashed changes
