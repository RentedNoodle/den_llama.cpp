<<<<<<< Updated upstream
<script lang="ts">
	import * as Sheet from '$lib/components/ui/sheet/index.js';
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';
	import { SIDEBAR_WIDTH_MOBILE } from './constants.js';
	import { useSidebar } from './context.svelte.js';

	let {
		ref = $bindable(null),
		side = 'left',
		variant = 'sidebar',
		collapsible = 'offcanvas',
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
		side?: 'left' | 'right';
		variant?: 'sidebar' | 'floating' | 'inset';
		collapsible?: 'offcanvas' | 'icon' | 'none';
	} = $props();

	const sidebar = useSidebar();
</script>

{#if collapsible === 'none'}
	<div
		class={cn(
			'flex h-full w-(--sidebar-width) flex-col bg-sidebar text-sidebar-foreground',
			className
		)}
		bind:this={ref}
		{...restProps}
	>
		{@render children?.()}
	</div>
{:else if sidebar.isMobile}
	<Sheet.Root bind:open={() => sidebar.openMobile, (v) => sidebar.setOpenMobile(v)} {...restProps}>
		<Sheet.Content
			data-sidebar="sidebar"
			data-slot="sidebar"
			data-mobile="true"
			class="z-99999 w-(--sidebar-width) bg-sidebar p-0 text-sidebar-foreground sm:z-99 [&>button]:hidden"
			style="--sidebar-width: {SIDEBAR_WIDTH_MOBILE};"
			{side}
		>
			<Sheet.Header class="sr-only">
				<Sheet.Title>Sidebar</Sheet.Title>
				<Sheet.Description>Displays the mobile sidebar.</Sheet.Description>
			</Sheet.Header>
			<div class="flex h-full w-full flex-col">
				{@render children?.()}
			</div>
		</Sheet.Content>
	</Sheet.Root>
{:else}
	<div
		bind:this={ref}
		class="group peer hidden text-sidebar-foreground md:block"
		data-state={sidebar.state}
		data-collapsible={sidebar.state === 'collapsed' ? collapsible : ''}
		data-variant={variant}
		data-side={side}
		data-slot="sidebar"
	>
		<!-- This is what handles the sidebar gap on desktop -->
		<div
			data-slot="sidebar-gap"
			class={cn(
				'relative w-(--sidebar-width) bg-transparent transition-[width] duration-200 ease-linear',
				'group-data-[collapsible=offcanvas]:w-0',
				'group-data-[side=right]:rotate-180',
				variant === 'floating' || variant === 'inset'
					? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)'
			)}
		></div>
		<div
			data-slot="sidebar-container"
			class={cn(
				'fixed inset-y-0 z-999 hidden h-svh w-(--sidebar-width) transition-[left,right,width] duration-200 ease-linear md:z-0 md:flex',
				side === 'left'
					? 'left-0 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)]'
					: 'right-0 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]',
				// Adjust the padding for floating and inset variants.
				variant === 'floating' || variant === 'inset'
					? 'p-2 group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)',
				className
			)}
			{...restProps}
		>
			<div
				data-sidebar="sidebar"
				data-slot="sidebar-inner"
				class="flex h-full w-full flex-col bg-sidebar group-data-[variant=floating]:rounded-lg group-data-[variant=floating]:border group-data-[variant=floating]:border-sidebar-border group-data-[variant=floating]:shadow-sm"
			>
				{@render children?.()}
			</div>
		</div>
	</div>
{/if}
=======
<script lang="ts">
	import { cn, type WithElementRef } from '$lib/components/ui/utils.js';
	import type { HTMLAttributes } from 'svelte/elements';
	import { SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH } from './constants.js';
	import { useSidebar } from './context.svelte.js';
	import { remToPx } from '$lib/utils';

	let {
		ref = $bindable(null),
		side = 'left',
		variant = 'sidebar',
		collapsible = 'offcanvas',
		class: className,
		children,
		...restProps
	}: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
		side?: 'left' | 'right';
		variant?: 'sidebar' | 'floating' | 'inset';
		collapsible?: 'offcanvas' | 'icon' | 'none';
	} = $props();

	const sidebar = useSidebar();

	function handleResizePointerDown(e: PointerEvent) {
		if (sidebar.isMobile) return;
		e.preventDefault();

		const target = e.currentTarget as HTMLElement;
		target.setPointerCapture(e.pointerId);

		const minPx = remToPx(SIDEBAR_MIN_WIDTH);
		const maxPx = remToPx(SIDEBAR_MAX_WIDTH);

		sidebar.isResizing = true;

		function onPointerMove(ev: PointerEvent) {
			const newWidth = side === 'left' ? ev.clientX : window.innerWidth - ev.clientX;
			const clamped = Math.min(maxPx, Math.max(minPx, newWidth));
			sidebar.sidebarWidth = `${clamped}px`;
		}

		function onPointerUp() {
			sidebar.isResizing = false;
			target.removeEventListener('pointermove', onPointerMove);
			target.removeEventListener('pointerup', onPointerUp);
		}

		target.addEventListener('pointermove', onPointerMove);
		target.addEventListener('pointerup', onPointerUp);
	}
</script>

{#if collapsible === 'none'}
	<div
		class={cn(
			'flex h-full w-(--sidebar-width) flex-col bg-sidebar text-sidebar-foreground',
			className
		)}
		bind:this={ref}
		{...restProps}
	>
		{@render children?.()}
	</div>
{:else}
	<div
		bind:this={ref}
		class="group peer block text-sidebar-foreground"
		data-state={sidebar.state}
		data-collapsible={sidebar.state === 'collapsed' ? collapsible : ''}
		data-variant={variant}
		data-side={side}
		data-slot="sidebar"
	>
		<!-- This is what handles the sidebar gap on desktop -->
		<div
			data-slot="sidebar-gap"
			class={cn(
				'relative bg-transparent transition-[width] duration-200 ease-linear',
				sidebar.isResizing && '!duration-0',
				'w-0',
				variant === 'floating'
					? 'md:w-[calc(var(--sidebar-width)+0.75rem)]'
					: 'md:w-(--sidebar-width)',
				'md:group-data-[collapsible=offcanvas]:w-0',
				'group-data-[side=right]:rotate-180',
				variant === 'floating' || variant === 'inset'
					? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)'
			)}
		></div>

		<div
			data-slot="sidebar-container"
			class={cn(
				'fixed inset-y-0 z-[900] flex w-[calc(100dvw-1.5rem)] duration-200 ease-linear md:z-0 md:w-(--sidebar-width)',
				'group-data-[collapsible=offcanvas]:pointer-events-none md:group-data-[collapsible=offcanvas]:pointer-events-auto',
				sidebar.isResizing && '!duration-0',
				variant === 'floating'
					? [
							'transition-[left,right,width,opacity]',
							side === 'left'
								? 'left-3 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-0.775)] group-data-[collapsible=offcanvas]:opacity-0'
								: 'right-3 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-0.775)] group-data-[collapsible=offcanvas]:opacity-0',
							'my-3 overflow-hidden rounded-3xl border border-sidebar-border shadow-md'
						]
					: [
							'h-svh transition-[left,right,width]',
							side === 'left'
								? 'left-0 group-data-[collapsible=offcanvas]:left-[calc(var(--sidebar-width)*-1)]'
								: 'right-0 group-data-[collapsible=offcanvas]:right-[calc(var(--sidebar-width)*-1)]'
						],
				// Adjust the padding for inset variant.
				variant === 'inset'
					? 'p-2 group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
					: variant === 'floating'
						? 'group-data-[collapsible=icon]:w-[calc(var(--sidebar-width-icon)+(--spacing(4))+2px)]'
						: 'group-data-[collapsible=icon]:w-(--sidebar-width-icon)',
				className
			)}
			style={variant === 'floating' ? 'height: calc(100dvh - 1.5rem);' : undefined}
			{...restProps}
		>
			<div
				data-sidebar="sidebar"
				data-slot="sidebar-inner"
				class="flex h-full w-full flex-col bg-sidebar"
			>
				{@render children?.()}
			</div>
			<!-- Resize handle -->
			{#if side === 'left'}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-slot="sidebar-resize-handle"
					class="absolute inset-y-0 right-0 z-50 hidden w-1.5 cursor-ew-resize touch-none select-none hover:bg-sidebar-border/50 active:bg-sidebar-border md:block"
					class:bg-sidebar-border={sidebar.isResizing}
					onpointerdown={handleResizePointerDown}
				></div>
			{:else}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					data-slot="sidebar-resize-handle"
					class="absolute inset-y-0 left-0 z-50 hidden w-1.5 cursor-ew-resize touch-none select-none hover:bg-sidebar-border/50 active:bg-sidebar-border md:block"
					class:bg-sidebar-border={sidebar.isResizing}
					onpointerdown={handleResizePointerDown}
				></div>
			{/if}
		</div>
	</div>
{/if}
>>>>>>> Stashed changes
